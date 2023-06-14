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
    stereo->setSubpixel(true);
    stereo->setExtendedDisparity(true);
    stereo->setRectifyEdgeFillColor(0);
    // stereo->setAlphaScaling(0.0);
    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_DENSITY);
    stereo->initialConfig.setMedianFilter(dai::MedianFilter::KERNEL_5x5);
    auto config = stereo->initialConfig.get();
    config.costMatching.disparityWidth = dai::StereoDepthConfig::CostMatching::DisparityWidth::DISPARITY_64;
    stereo->initialConfig.set(config);

    manip->setKeepAspectRatio(false);
    manip->initialConfig.setResize(320, 200);

    superPointNetwork->setBlobPath(nnPath);
    superPointNetwork->setNumInferenceThreads(1);
    superPointNetwork->setNumNCEPerInferenceThread(2);
    superPointNetwork->input.setBlocking(false);

    monoLeft->out.link(stereo->left);
    monoRight->out.link(stereo->right);
    stereo->disparity.link(xoutDisp->input);
    stereo->rectifiedLeft.link(xoutLeft->input);
    stereo->rectifiedLeft.link(manip->inputImage);
    manip->out.link(superPointNetwork->input);
    superPointNetwork->out.link(xoutNN->input);

    dai::Device device(pipeline);

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

    while (true)
    {
        auto left = leftQueue->get<dai::ImgFrame>();
        auto disparity = dispQueue->get<dai::ImgFrame>();
        while (disparity->getSequenceNum() < left->getSequenceNum())
            disparity = dispQueue->get<dai::ImgFrame>();
        auto superPoint = superPointQueue->get<dai::NNData>();
        while (superPoint->getSequenceNum() < left->getSequenceNum())
            superPoint = superPointQueue->get<dai::NNData>();

        cv::Mat mono, disp, heatmap, blended;
        cv::cvtColor(left->getFrame(), mono, cv::COLOR_GRAY2BGR);
        disparity->getFrame().convertTo(disp, CV_8UC1, 255.0 / 1001);
        cv::applyColorMap(disp, disp, cv::COLORMAP_TURBO);
        cv::resize(fromPlanarFp16(superPoint->getLayerFp16("heatmap"), 320, 200, confThresh), heatmap, cv::Size(1280, 800));
        cv::cvtColor(heatmap, heatmap, cv::COLOR_GRAY2BGR);
        cv::subtract(255, heatmap, blended);
        cv::addWeighted(mono.mul(blended, 1.0 / 255), 1, disp.mul(heatmap, 1.0 / 255), 1, 0, blended);
        cv::imshow("SuperPoint", blended);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q')
            return 0;
    }

    return 0;
}