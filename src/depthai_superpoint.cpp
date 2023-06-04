#include <depthai/depthai.hpp>

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
    monoRight->setBoardSocket(dai::CameraBoardSocket::RIGHT);
    monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_800_P);

    stereo->setDepthAlign(dai::StereoDepthProperties::DepthAlign::RECTIFIED_LEFT);
    stereo->setSubpixel(true);
    stereo->setSubpixelFractionalBits(4);
    stereo->setExtendedDisparity(false);
    stereo->setRectifyEdgeFillColor(0);
    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_DENSITY);
    stereo->initialConfig.setMedianFilter(dai::MedianFilter::KERNEL_5x5);
    auto config = stereo->initialConfig.get();
    config.costMatching.disparityWidth = dai::StereoDepthConfig::CostMatching::DisparityWidth::DISPARITY_64;
    config.costMatching.enableCompanding = true;
    stereo->initialConfig.set(config);

    manip->initialConfig.setResize(320, 200);

    superPointNetwork->setBlobPath(nnPath);
    superPointNetwork->setNumInferenceThreads(2);
    superPointNetwork->input.setBlocking(false);

    monoLeft->out.link(stereo->left);
    monoRight->out.link(stereo->right);
    stereo->rectifiedLeft.link(manip->inputImage);
    manip->out.link(superPointNetwork->input);
    superPointNetwork->passthrough.link(xoutLeft->input);
    superPointNetwork->out.link(xoutNN->input);

    dai::Device device(pipeline);

    auto leftQueue = device.getOutputQueue("rectified_left", 8, false);
    auto superPointQueue = device.getOutputQueue("nn", 8, false);

    while (true)
    {
        auto left = leftQueue->get<dai::ImgFrame>();
        auto superPoint = superPointQueue->get<dai::NNData>();
        cv::imshow("left", left->getFrame());

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q')
            return 0;
    }

    return 0;
}