import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;

import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_WIDTH;
import static org.opencv.highgui.HighGui.destroyAllWindows;
import static org.opencv.videoio.Videoio.*;

//Special thanks for yquemener for writing the original camera stream class
public class CameraStreamer extends Thread {

    public Runnable camStreamRunnable;
    private int videoCaptureIndex = 0;
//    private String videoCaptureIndex = "C:\\Users\\ChooWilson\\Desktop\\koreacrowd.mp4";

    private int videoCaptureFPS = 30;
    private VideoCapture cap = null;
    private NativeImageLoader nil = null;
    private final int videoCaptureWidth = 1280;
    private final int videoCaptureHeight = 720;
    private final int outputWidth = 416;
    private final int outputHeight = 416;

    private OpenCVFrameConverter.ToMat converterMatToFrame;
    private Java2DFrameConverter converterFrameToBufferedImage;


    public void startCapture() {

//        converterFrameToBufferedImage = new Java2DFrameConverter();
//        converterMatToFrame = new OpenCVFrameConverter.ToMat();

        if (this.cap == null) {
            this.cap = new VideoCapture();
        }
        if (this.nil == null) {
            this.nil = new NativeImageLoader(
                    this.outputHeight,
                    this.outputWidth,
                    3);
        }
        this.cap.open(this.videoCaptureIndex);
        this.cap.set(CAP_PROP_FRAME_WIDTH, this.videoCaptureWidth);
        this.cap.set(CAP_PROP_FRAME_HEIGHT, this.videoCaptureHeight);
        this.cap.set(CAP_PROP_FPS, this.videoCaptureFPS);

        // Disable autofocus
        // If the picture is blurry for closeup objects, you may want to change this
        this.cap.set(CAP_PROP_AUTOFOCUS, 0);
        this.cap.set(CAP_PROP_FOCUS, 0);
        this.start();
    }

    public Mat getCVFrame() {
        Mat cvFrame = new Mat();
        // For some reasons, OpenCV capture needs that to prevent memory leaks, explicitly calling the garbage collector
        System.gc();
        synchronized (cap) {
            cap.retrieve(cvFrame);
        }
        assert (this.outputHeight <= this.videoCaptureHeight);
        assert (this.outputWidth <= this.videoCaptureWidth);
        assert (cvFrame != null);
        assert (!cvFrame.empty());
        return cvFrame;
    }

    public INDArray getNDArray(Mat frame) throws IOException {
        return nil.asMatrix(frame).div(255.0);
    }

    //    Destroy the VideoCapture properly
    public void stopCapture() {
        if (!cap.isNull())
            this.cap.release();
        destroyAllWindows();
    }

    //    Run VideoCapture as an independent thread
    @Override
    public void run() {
        while (isAlive()) {
            synchronized (cap) {
                this.cap.grab();
            }
            try {
                camStreamRunnable.run();
            } catch (Exception e) {
                System.err.println("Camera Stream Thread threw an exception. Error message: " + e.toString());
            }
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                e.printStackTrace();
                stopCapture();
            }
        }
        stopCapture();
    }

    //    public BufferedImage getBufferedImage() throws IOException {
    //        Frame f = converterMatToFrame.convert(getCVFrame());
    //        return converterFrameToBufferedImage.convert(f);
    //    }
}
