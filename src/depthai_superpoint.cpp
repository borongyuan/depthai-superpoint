#include <depthai/depthai.hpp>

void updateConfThresh(int percentConfThresh, void *conf_thresh)
{
    *((float *)conf_thresh) = float(percentConfThresh) / 100.f;
}

cv::Mat fromPlanarFp16(const std::vector<float> &data, int w, int h, float conf_thresh)
{
    cv::Mat frame = cv::Mat(h, w, CV_8UC1);
    for (int i = 0; i < w * h; i++)
        frame.data[i] = data.data()[i] < conf_thresh ? 0 : 255;
    return frame;
}

int main(int argc, char **argv)
{
    auto nnPath = std::string(argv[1]);
    std::cout << "Using blob at path: " << nnPath.c_str() << std::endl;

    bool res = false;
    dai::DeviceInfo info;
    std::tie(res, info) = dai::Device::getFirstAvailableDevice();

    if (!res)
    {
        std::cout << "No devices found" << std::endl;
        return -1;
    }

    dai::Pipeline pipeline;

    auto monoLeft = pipeline.create<dai::node::MonoCamera>();
    auto monoRight = pipeline.create<dai::node::MonoCamera>();
    auto stereo = pipeline.create<dai::node::StereoDepth>();
    auto manip = pipeline.create<dai::node::ImageManip>();
    auto superPointNetwork = pipeline.create<dai::node::NeuralNetwork>();

    auto xoutLeft = pipeline.create<dai::node::XLinkOut>();
    auto xoutDisp = pipeline.create<dai::node::XLinkOut>();
    auto xoutNN = pipeline.create<dai::node::XLinkOut>();

    xoutLeft->setStreamName("rectified_left");
    xoutDisp->setStreamName("disparity");
    xoutNN->setStreamName("nn");

    monoLeft->setCamera("left");
    monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_800_P);
    monoLeft->setFps(15);
    monoRight->setCamera("right");
    monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_800_P);
    monoRight->setFps(15);

    stereo->setDepthAlign(dai::StereoDepthProperties::DepthAlign::RECTIFIED_LEFT);
    stereo->setExtendedDisparity(true);
    stereo->setRectifyEdgeFillColor(0);
    // stereo->setAlphaScaling(0.0);
    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_DENSITY);
    stereo->initialConfig.setMedianFilter(dai::MedianFilter::KERNEL_5x5);

    manip->setKeepAspectRatio(false);
    manip->setMaxOutputFrameSize(320 * 200);
    manip->initialConfig.setResize(320, 200);

    superPointNetwork->setBlobPath(nnPath);
    superPointNetwork->setNumInferenceThreads(2);
    superPointNetwork->setNumNCEPerInferenceThread(1);
    superPointNetwork->input.setBlocking(false);

    monoLeft->out.link(stereo->left);
    monoRight->out.link(stereo->right);
    if (info.protocol == X_LINK_TCP_IP)
    {
        auto stereoEnc = pipeline.create<dai::node::VideoEncoder>();
        auto leftEnc = pipeline.create<dai::node::VideoEncoder>();
        stereoEnc->setDefaultProfilePreset(15, dai::VideoEncoderProperties::Profile::MJPEG);
        leftEnc->setDefaultProfilePreset(15, dai::VideoEncoderProperties::Profile::MJPEG);
        stereo->disparity.link(stereoEnc->input);
        stereo->rectifiedLeft.link(leftEnc->input);
        stereoEnc->bitstream.link(xoutDisp->input);
        leftEnc->bitstream.link(xoutLeft->input);
    }
    else
    {
        stereo->disparity.link(xoutDisp->input);
        stereo->rectifiedLeft.link(xoutLeft->input);
    }
    stereo->rectifiedLeft.link(manip->inputImage);
    manip->out.link(superPointNetwork->input);
    superPointNetwork->out.link(xoutNN->input);

    dai::Device device(pipeline, info);

    auto leftQueue = device.getOutputQueue("rectified_left", 8, false);
    auto dispQueue = device.getOutputQueue("disparity", 8, false);
    auto superPointQueue = device.getOutputQueue("nn", 8, false);

    std::vector<std::tuple<std::string, int, int>> irDrivers = device.getIrDrivers();
    if (!irDrivers.empty())
    {
        device.setIrLaserDotProjectorBrightness(0);
        device.setIrFloodLightBrightness(1500);
    }

    float confThresh = 0.01f;
    int defaultValue = (int)(confThresh * 100);
    cv::namedWindow("SuperPoint");
    cv::createTrackbar("Detector confidence threshold %", "SuperPoint", &defaultValue, 100, updateConfThresh, &confThresh);

    cv::Mat mono, disp, heatmap, blended;
    while (true)
    {
        auto left = leftQueue->get<dai::ImgFrame>();
        auto disparity = dispQueue->get<dai::ImgFrame>();
        while (disparity->getSequenceNum() < left->getSequenceNum())
            disparity = dispQueue->get<dai::ImgFrame>();
        auto superPoint = superPointQueue->get<dai::NNData>();
        while (superPoint->getSequenceNum() < left->getSequenceNum())
            superPoint = superPointQueue->get<dai::NNData>();

        if (info.protocol == X_LINK_TCP_IP)
        {
            cv::imdecode(left->getData(), cv::IMREAD_GRAYSCALE, &mono);
            cv::imdecode(disparity->getData(), cv::IMREAD_GRAYSCALE, &disp);
        }
        else
        {
            mono = left->getFrame();
            disp = disparity->getFrame();
        }
        cv::cvtColor(mono, mono, cv::COLOR_GRAY2BGR);
        cv::applyColorMap(disp, disp, cv::COLORMAP_TURBO);
        cv::resize(fromPlanarFp16(superPoint->getLayerFp16("heatmap"), 320, 200, confThresh), heatmap, cv::Size(1280, 800));
        cv::cvtColor(heatmap, heatmap, cv::COLOR_GRAY2BGR);
        cv::subtract(255, heatmap, blended);
        cv::addWeighted(mono.mul(blended, 1.0 / 255), 1, disp.mul(heatmap, 1.0 / 255), 1, 0, blended);
        cv::imshow("SuperPoint", blended);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q')
        {
            device.close();
            return 0;
        }
    }

    return 0;
}