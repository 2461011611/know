#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Float32.h>
#include <vector>
#include <signal.h>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include "HCUsbSDK.h"

#define MAX_USB_DEV_LEN 32
#define MAX_RETRY 3



// 创建于 2026-03    海康微影热成像USB-TM52设备    实现：取流和温度数据


// ============================================================
// 分辨率与帧大小常量定义
// ============================================================

// 1. 物理传输分辨率 (触发相机进入高级测温流模式的关键)
static const int PHYSICAL_WIDTH  = 304;
static const int PHYSICAL_HEIGHT = 331;
static const int TARGET_FPS      = 50; // 保证 USB 传输稳定性

// 2. 逻辑热成像分辨率 (相机实际输出的有效数据尺寸)
static const int THERMAL_WIDTH   = 256;
static const int THERMAL_HEIGHT  = 192;

// 3. 数据段大小常量
static const int TEMP_DATA_SIZE = THERMAL_WIDTH * THERMAL_HEIGHT * 2; // 16位测温矩阵 (98304 字节)
static const int YUV_DATA_SIZE  = THERMAL_WIDTH * THERMAL_HEIGHT * 2; // YUY2 图像数据 (98304 字节)

// ============================================================
// 全局变量
// ============================================================
static bool  g_bRunning     = true;
static LONG  g_userID       = -1;
static LONG  g_streamHandle = -1;
static ros::Publisher g_img_pub;
static ros::Publisher g_temp_pub;

void signalHandler(int sig) {
    ROS_INFO("Received signal %d, shutting down...", sig);
    g_bRunning = false;
}

// ============================================================
// 辅助函数: YUYV 转 BGR
// ============================================================
void YUYVToBGR8(const uint8_t* yuyv, int width, int height, uint8_t* bgr) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; x += 2) {
            int idx = y * width * 2 + x * 2;
            uint8_t y0 = yuyv[idx];
            uint8_t u  = yuyv[idx + 1];
            uint8_t y1 = yuyv[idx + 2];
            uint8_t v  = yuyv[idx + 3];

            int c0 = y0 - 16, c1 = y1 - 16;
            int d  = u - 128, e  = v - 128;

            auto clamp = [](int v) -> uint8_t {
                return v < 0 ? 0 : (v > 255 ? 255 : (uint8_t)v);
            };

            int outIdx = (y * width + x) * 3;
            bgr[outIdx + 0] = clamp((298 * c0 + 516 * d + 128) >> 8);
            bgr[outIdx + 1] = clamp((298 * c0 - 100 * d - 208 * e + 128) >> 8);
            bgr[outIdx + 2] = clamp((298 * c0 + 409 * e + 128) >> 8);
            bgr[outIdx + 3] = clamp((298 * c1 + 516 * d + 128) >> 8);
            bgr[outIdx + 4] = clamp((298 * c1 - 100 * d - 208 * e + 128) >> 8);
            bgr[outIdx + 5] = clamp((298 * c1 + 409 * e + 128) >> 8);
        }
    }
}

// ============================================================
// 终极鲁棒的回调函数 (彻底无视头部结构的对齐问题)
// ============================================================
void CALLBACK streamCallback(LONG userID, USB_FRAME_INFO* pFrameInfo, void* pUser) {
    if (!g_bRunning) return;
    if (!pFrameInfo || !pFrameInfo->pBuf || pFrameInfo->dwBufSize == 0) return;

    const uint8_t* buf = pFrameInfo->pBuf;
    uint32_t size = pFrameInfo->dwBufSize;

    // 1. 过滤掉启动时产生的残缺过渡帧
    if (size < TEMP_DATA_SIZE + YUV_DATA_SIZE) {
        ROS_WARN_THROTTLE(2.0, "Ignoring transient frame, size: %u bytes", size);
        return;
    }

    // 2. 动态计算头部大小 (完全自适应 201248、201244 或 196608 等任何情况)
    uint32_t header_size = size - TEMP_DATA_SIZE - YUV_DATA_SIZE;
    
    // 3. 精准定位到 RAW 和 YUV 数据的开头
    const uint8_t* pRawData = buf + header_size;
    const uint8_t* pYUVData = buf + header_size + TEMP_DATA_SIZE;

    float maxTmp = -273.15f;

    // 4. 尝试利用 SDK 自带的结构体强转解析 (如果是标准的带头包)
    if (header_size >= sizeof(USB_THERMAL_STREAM_TEMP_HOT)) {
        USB_THERMAL_STREAM_TEMP_HOT* pHeader = (USB_THERMAL_STREAM_TEMP_HOT*)buf;
        maxTmp = pHeader->struRTDataUpload.fMaxTmp;
    }

    // 5. 核心保底策略：如果头部解析出乱码 (例如极小值 1e-45 或极大值)，直接暴力遍历 RAW 矩阵！
    if (maxTmp < -40.0f || maxTmp > 1500.0f) {
        const int16_t* raw_matrix = (const int16_t*)pRawData;
        int16_t max_raw = -32768;
        for (int i = 0; i < THERMAL_WIDTH * THERMAL_HEIGHT; ++i) {
            if (raw_matrix[i] > max_raw) {
                max_raw = raw_matrix[i];
            }
        }
        // 海康的 RAW 数据通常换算比例是 10.0 (温度 = raw / 10.0)
        maxTmp = (float)max_raw / 10.0f;
    }

    // 6. 发布温度
    std_msgs::Float32 tempMsg;
    tempMsg.data = maxTmp;
    g_temp_pub.publish(tempMsg);

    // 每秒打印一次成功信息
    // ROS_INFO_STREAM_THROTTLE(1.0, "Parsed OK! Frame: " << size 
    //                            << " bytes, Header: " << header_size 
    //                            << " bytes, Max Temp: " << maxTmp << "°C");

    // 7. 解析 YUV 图像并发布
    std::vector<uint8_t> bgr_data(THERMAL_WIDTH * THERMAL_HEIGHT * 3);
    YUYVToBGR8(pYUVData, THERMAL_WIDTH, THERMAL_HEIGHT, bgr_data.data());

    sensor_msgs::Image img_msg;
    img_msg.header.stamp = ros::Time::now();
    img_msg.header.frame_id = "thermal_camera";
    img_msg.height = THERMAL_HEIGHT;
    img_msg.width = THERMAL_WIDTH;
    img_msg.encoding = sensor_msgs::image_encodings::BGR8;
    img_msg.step = THERMAL_WIDTH * 3;
    img_msg.data.assign(bgr_data.begin(), bgr_data.end());
    g_img_pub.publish(img_msg);
}

// ============================================================
// 设备查找 & 登录
// ============================================================
int FindDevice(const USB_DEVICE_INFO* devList, int count) {
    for (int i = 0; i < count; ++i) {
        if (devList[i].dwVID == 0x2BDF && devList[i].dwPID == 0x0102)
            return i;
    }
    return -1;
}

LONG LoginDevice(const USB_DEVICE_INFO* pDevInfo) {
    USB_USER_LOGIN_INFO loginInfo = {0};
    loginInfo.dwSize    = sizeof(loginInfo);
    loginInfo.dwTimeout = 5000;
    loginInfo.dwVID     = pDevInfo->dwVID;
    loginInfo.dwPID     = pDevInfo->dwPID;
    strcpy(loginInfo.szSerialNumber, (const char*)pDevInfo->szSerialNumber);

    const struct { int mode; const char* user; const char* pwd; } trials[] = {
        {0, "",      ""     },
        {0, "admin", ""     },
        {0, "admin", "12345"},
        {1, "",      ""     }
    };

    for (int i = 0; i < (int)(sizeof(trials)/sizeof(trials[0])); ++i) {
        loginInfo.byLoginMode = trials[i].mode;
        strcpy(loginInfo.szUserName, trials[i].user);
        strcpy(loginInfo.szPassword, trials[i].pwd);
        USB_DEVICE_REG_RES regRes;
        LONG userID = USB_Login(&loginInfo, &regRes);
        if (userID != -1) return userID;
        usleep(200000);
    }
    return -1;
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv) {
    ros::init(argc, argv, "hk_thera", ros::init_options::NoSigintHandler);
    ros::NodeHandle nh("~");
    signal(SIGINT, signalHandler);

    // ========== 1. 动态库路径 ==========
    USB_LOCAL_LOAD_PATH m_lib1, m_lib2;
    m_lib1.emType = ENUM_DLL_LIBUSB_PATH;
    strcpy((char*)m_lib1.byLoadPath, "/home/ysc/robot_dog/src/hk_thera/lib/libusb-1.0.so");
    m_lib2.emType = ENUM_DLL_LIBUVC_PATH;
    strcpy((char*)m_lib2.byLoadPath, "/home/ysc/robot_dog/src/hk_thera/lib/libuvc.so");
    USB_SetSDKLocalCfg(ENUM_LOCAL_CFG_TYPE_LOAD_PATH, (void*)&m_lib1);
    USB_SetSDKLocalCfg(ENUM_LOCAL_CFG_TYPE_LOAD_PATH, (void*)&m_lib2);

    // ========== 2. SDK 初始化 ==========
    if (!USB_Init()) {
        ROS_ERROR("USB_Init failed, error: %d", USB_GetLastError());
        return -1;
    }
    ROS_INFO("USB SDK Version: %d", USB_GetSDKVersion());

    // ========== 3. 日志 ==========
    const char* logPath = "/home/ysc/robot_dog/logs/hk_sdk/";
    system(("mkdir -p " + std::string(logPath)).c_str());
    if (USB_SetLogToFile(3, logPath, TRUE))
        ROS_INFO("SDK log -> %s", logPath);

    // ========== 4. 枚举设备 ==========
    int devCount = 0;
    for (int i = 0; i < MAX_RETRY; ++i) {
        devCount = USB_GetDeviceCount();
        if (devCount > 0) break;
        usleep(500000);
    }
    if (devCount <= 0) {
        ROS_ERROR("No USB device found");
        USB_Cleanup(); return -1;
    }

    std::vector<USB_DEVICE_INFO> devList(devCount);
    if (!USB_EnumDevices(devCount, devList.data())) {
        ROS_ERROR("EnumDevices failed: %d", USB_GetLastError());
        USB_Cleanup(); return -1;
    }

    int devIndex = FindDevice(devList.data(), devCount);
    if (devIndex < 0) {
        ROS_ERROR("Thermal device not found");
        USB_Cleanup(); return -1;
    }

    // ========== 5. 登录 ==========
    g_userID = LoginDevice(&devList[devIndex]);
    if (g_userID == -1) {
        ROS_ERROR("Login failed");
        USB_Cleanup(); return -1;
    }
    ROS_INFO("Login OK, UserID=%d", g_userID);

    // ========== 6. 设置视频参数 (物理传输载体参数 304x331) ==========
    ROS_INFO("Setting video param: %dx%d @ %dfps YUY2", PHYSICAL_WIDTH, PHYSICAL_HEIGHT, TARGET_FPS);
    USB_VIDEO_PARAM videoParam = {0};
    videoParam.dwVideoFormat = USB_STREAM_YUY2;
    videoParam.dwWidth       = PHYSICAL_WIDTH;
    videoParam.dwHeight      = PHYSICAL_HEIGHT;
    videoParam.dwFramerate   = TARGET_FPS;

    USB_CONFIG_INPUT_INFO  inInfo  = {0};
    USB_CONFIG_OUTPUT_INFO outInfo = {0};
    inInfo.lpInBuffer    = &videoParam;
    inInfo.dwInBufferSize = sizeof(videoParam);
    if (!USB_SetDeviceConfig(g_userID, USB_SET_VIDEO_PARAM, &inInfo, &outInfo)) {
        ROS_WARN("SetVideoParam failed (%d)", USB_GetLastError());
    } else {
        ROS_INFO("Video param set OK");
    }

    // ========== 7. 设置热成像参数 (必须用逻辑分辨率 256x192) ==========
    USB_THERMAL_STREAM_PARAM thermalParam;
    memset(&thermalParam, 0, sizeof(thermalParam));
    thermalParam.dwSize = sizeof(thermalParam);
    thermalParam.byVideoCodingType = 8;        // 码流8
    thermalParam.dwWidth = THERMAL_WIDTH;      // 必须是 256
    thermalParam.dwHeight = THERMAL_HEIGHT;    // 必须是 192
    thermalParam.dwFrameRate = TARGET_FPS;     // 必须是 50

    inInfo.lpInBuffer = &thermalParam;
    inInfo.dwInBufferSize = sizeof(thermalParam);
    if (!USB_SetDeviceConfig(g_userID, USB_SET_THERMAL_STREAM_PARAM, &inInfo, &outInfo)) {
        ROS_WARN("Set Thermal Stream Param failed (%d)", USB_GetLastError());
    }

    // ========== 8. 启动码流 ==========
    USB_STREAM_CALLBACK_PARAM streamParam = {0};
    streamParam.dwSize           = sizeof(streamParam);
    streamParam.dwStreamType     = USB_STREAM_YUY2;
    streamParam.funcStreamCallBack = streamCallback;
    streamParam.pUser            = NULL;
    streamParam.bUseAudio        = 0;

    g_streamHandle = USB_StartStreamCallback(g_userID, &streamParam);
    if (g_streamHandle == -1) {
        ROS_ERROR("StartStreamCallback failed: %d", USB_GetLastError());
        USB_Logout(g_userID); USB_Cleanup(); return -1;
    }
    ROS_INFO("Stream started OK");

    // ========== 9. 注册 ROS 发布者 ==========
    g_img_pub  = nh.advertise<sensor_msgs::Image>("thermal_image", 10);
    g_temp_pub = nh.advertise<std_msgs::Float32>("thermal_max_temperature", 10);

    // ========== 10. 主循环 ==========
    ros::Rate rate(30);
    while (ros::ok() && g_bRunning) {
        ros::spinOnce();
        rate.sleep();
    }

    // ========== 11. 清理 ==========
    if (g_streamHandle != -1) USB_StopChannel(g_userID, g_streamHandle);
    if (g_userID != -1)       USB_Logout(g_userID);
    USB_Cleanup();
    ROS_INFO("Thermal node stopped");
    return 0;
}