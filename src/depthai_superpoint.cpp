#include <depthai/depthai.hpp>

cv::Mat fromPlanarFp16(const std::vector<float> &data, int w, int h, float mean, float scale)
{
    cv::Mat frame = cv::Mat(h, w, CV_8UC1);
    for (int i = 0; i < w * h; i++)
        frame.data[i] = (uint8_t)(data.data()[i] * scale + mean);
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
    auto xoutNN = pipeline.create<dai::node::XLinkOut>();

    xoutLeft->setStreamName("rectified_left");
    xoutNN->setStreamName("nn");

    monoLeft->setBoardSocket(dai::CameraBoardSocket::LEFT);
    monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_800_P);
    monoLeft->setFps(15);
    monoRight->setBoardSocket(dai::CameraBoardSocket::RIGHT);
    monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_800_P);
    monoRight->setFps(15);

    stereo->setDepthAlign(dai::StereoDepthProperties::DepthAlign::RECTIFIED_LEFT);
    stereo->setSubpixel(true);
    stereo->setSubpixelFractionalBits(4);
    stereo->setExtendedDisparity(false);
    stereo->setRectifyEdgeFillColor(0);
    // stereo->setAlphaScaling(0.0);
    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_DENSITY);
    stereo->initialConfig.setMedianFilter(dai::MedianFilter::KERNEL_5x5);
    auto config = stereo->initialConfig.get();
    config.costMatching.disparityWidth = dai::StereoDepthConfig::CostMatching::DisparityWidth::DISPARITY_64;
    config.costMatching.enableCompanding = true;
    stereo->initialConfig.set(config);

    manip->setKeepAspectRatio(false);
    manip->initialConfig.setResize(320, 200);

    superPointNetwork->setBlobPath(nnPath);
    superPointNetwork->setNumInferenceThreads(1);
    superPointNetwork->setNumNCEPerInferenceThread(2);
    superPointNetwork->input.setBlocking(false);

    monoLeft->out.link(stereo->left);
    monoRight->out.link(stereo->right);
    stereo->rectifiedLeft.link(xoutLeft->input);
    stereo->rectifiedLeft.link(manip->inputImage);
    manip->out.link(superPointNetwork->input);
    superPointNetwork->out.link(xoutNN->input);

    dai::Device device(pipeline);

    auto leftQueue = device.getOutputQueue("rectified_left", 8, false);
    auto superPointQueue = device.getOutputQueue("nn", 8, false);

    while (true)
    {
        auto left = leftQueue->get<dai::ImgFrame>();
        auto superPoint = superPointQueue->get<dai::NNData>();
        while (superPoint->getSequenceNum() < left->getSequenceNum())
            superPoint = superPointQueue->get<dai::NNData>();

        cv::Mat mono, heatmap, blended;
        cv::cvtColor(left->getFrame(), mono, cv::COLOR_GRAY2BGR);
        cv::resize(fromPlanarFp16(superPoint->getLayerFp16("heatmap"), 320, 200, 0.0, 255.0), heatmap, cv::Size(1280, 800));
        cv::applyColorMap(heatmap, heatmap, cv::COLORMAP_HOT);
        cv::addWeighted(mono, 1, heatmap, 1, 0, blended);
        cv::imshow("SuperPoint Heatmap", blended);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q')
            return 0;
    }

    return 0;
}