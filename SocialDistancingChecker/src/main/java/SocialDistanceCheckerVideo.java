import org.bytedeco.ffmpeg.global.avcodec;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;
import static org.nd4j.linalg.ops.transforms.Transforms.euclideanDistance;
import static org.nd4j.linalg.util.MathUtils.sigmoid;

//  Author: Choo Wilson(yptheangel)
// Special thanks to yquemener for writing yolov3 output interpreter

public class SocialDistanceCheckerVideo {

    private static ComputationGraph model;
    //        COCO has 80 classes
    private static int numClass = 80;
    private static int[][] anchors = {{10, 13}, {16, 30}, {33, 23}, {30, 61}, {62, 45}, {59, 119}, {116, 90}, {156, 198}, {373, 326}};
    private static int yolowidth = 608;
    private static int yoloheight = 608;

    public static void main(String[] args) throws InvalidKerasConfigurationException, IOException, UnsupportedKerasConfigurationException {

        int safeDistance = 80;
        model = KerasModelImport.importKerasModelAndWeights("C:\\Users\\ChooWilson\\Desktop\\yolov3_608_fixed.h5");
        model.init();
        //        System.out.println(model.summary());

        String videoPath = "C:\\Users\\choowilson\\Desktop\\crowd_1280x720.mp4";
        FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(videoPath);
        grabber.setFormat("mp4");
        grabber.start();

        FFmpegFrameRecorder recorder = new FFmpegFrameRecorder("output.mp4", 1280, 720, 0);
        recorder.setVideoCodec(avcodec.AV_CODEC_ID_MPEG4);
        recorder.setVideoBitrate(9000);
        recorder.setFormat("mp4");
        recorder.setVideoQuality(0); // maximum quality
        recorder.setFrameRate(15);
        recorder.start();

        OpenCVFrameConverter.ToMat frame2Mat = new OpenCVFrameConverter.ToMat();

        while (grabber.grab() != null) {
            Frame frame = grabber.grabImage();
            Mat opencvMat = frame2Mat.convert(frame);
            NativeImageLoader nil = new NativeImageLoader(yolowidth, yoloheight, 3);
            INDArray input = nil.asMatrix(opencvMat).div(255);

            List<DetectedObject> objs = getPredictedObjects(input);
            YoloUtils.nms(objs, 0.4);

            int w = opencvMat.cols();
            int h = opencvMat.rows();
            List<INDArray> centers = new ArrayList<>();
            List<INDArray> people = new ArrayList<>();

            int centerX;
            int centerY;

            for (DetectedObject obj : objs) {
                if (obj.getPredictedClass() == 0) {
                //            Scale the coordinates back to full size
                    centerX = (int) obj.getCenterX() * w / yolowidth;
                    centerY = (int) obj.getCenterY() * h / yoloheight;

                    circle(opencvMat, new Point(centerX, centerY), 3, new Scalar(0, 255, 0, 0), -1, 0, 0);
                    //            Draw bounding boxes on opencv mat
                    double[] xy1 = obj.getTopLeftXY();
                    double[] xy2 = obj.getBottomRightXY();
                    //            Scale the coordinates back to full size
                    xy1[0] = xy1[0] * w / yolowidth;
                    xy1[1] = xy1[1] * h / yoloheight;
                    xy2[0] = xy2[0] * w / yolowidth;
                    xy2[1] = xy2[1] * h / yoloheight;

                    //Draw bounding box
                    rectangle(opencvMat, new Point((int) xy1[0], (int) xy1[1]), new Point((int) xy2[0], (int) xy2[1]), new Scalar(0, 255, 0, 0), 2, LINE_8, 0);
                    centers.add(Nd4j.create(new float[]{(float) centerX, (float) centerY}));
                    people.add(Nd4j.create(new float[]{(float) xy1[0], (float) xy1[1], (float) xy2[0], (float) xy2[1]}));
                }
            }
            //        Calculate the euclidean distance between all pairs of center points
            for (int i = 0; i < centers.size(); i++) {
                for (int j = 0; j < centers.size(); j++) {
                    double distance = euclideanDistance(centers.get(i), centers.get(j));
                    if (distance < safeDistance && distance > 0) {
                        line(opencvMat, new Point(centers.get(i).getInt(0), centers.get(i).getInt(1)), new Point(centers.get(j).getInt(0), centers.get(j).getInt(1)), Scalar.RED, 2, 1, 0);

                        int xmin = people.get(i).getInt(0);
                        int ymin = people.get(i).getInt(1);
                        int xmax = people.get(i).getInt(2);
                        int ymax = people.get(i).getInt(3);

                        rectangle(opencvMat, new Point(xmin, ymin), new Point(xmax, ymax), Scalar.RED, 2, LINE_8, 0);
                        circle(opencvMat, new Point(centers.get(i).getInt(0), centers.get(i).getInt(1)), 3, Scalar.RED, -1, 0, 0);
                    }
                }
            }

            putText(opencvMat, String.format("Number of people: %d", people.size()), new Point(10, 25), 4, 0.8, Scalar.BLUE, 2, LINE_8, false);
            recorder.record(frame2Mat.convert(opencvMat));
            imshow("Social Distancing Checker", opencvMat);
            //    Press Esc key to quit
            if (waitKey(33) == 27) {
                recorder.stop();
                destroyAllWindows();
                break;
            }
        }
        recorder.stop();
    }

    public static List<DetectedObject> getPredictedObjects(INDArray input) {

        INDArray[] outputs = model.output(input);

        List<DetectedObject> out = new ArrayList();
        float detectionThreshold = 0.6f;
        // Each cell had information for 3 boxes
        int[] boxOffsets = {0, numClass + 5, (numClass + 5) * 2};

        for (int layerNum = 0; layerNum < 3; layerNum++) {
            long gridWidth = outputs[layerNum].shape()[2];
            long gridHeight = outputs[layerNum].shape()[3];
            float cellWidth = yolowidth / gridWidth;
            float cellHeight = yoloheight / gridHeight;

            for (int i = 0; i < gridHeight; i++) {
                for (int j = 0; j < gridWidth; j++) {
                    float centerX;
                    float centerY;
                    float width;
                    float height;
                    int anchorInd;

                    for (int k = 0; k < 3; k++) {
                        float prob = outputs[layerNum].getFloat(new int[]{0, boxOffsets[k] + 4, i, j});
                        if (prob > detectionThreshold) {
                            INDArray classes_scores = outputs[layerNum].get(
                                    point(0),
                                    NDArrayIndex.interval(boxOffsets[k] + 5, boxOffsets[k] + numClass + 5),
                                    point(i),
                                    point(j));
                            centerX = outputs[layerNum].getFloat(new int[]{0, boxOffsets[k] + 0, i, j});
                            centerY = outputs[layerNum].getFloat(new int[]{0, boxOffsets[k] + 1, i, j});
                            width = outputs[layerNum].getFloat(new int[]{0, boxOffsets[k] + 2, i, j});
                            height = outputs[layerNum].getFloat(new int[]{0, boxOffsets[k] + 3, i, j});
                            anchorInd = (2 - layerNum) * 3 + k;

                            centerX = (float) ((sigmoid(centerX) + j) * cellWidth);
                            centerY = (float) ((sigmoid(centerY) + i) * cellHeight);
                            width = (float) (Math.exp(width)) * anchors[anchorInd][0];
                            height = (float) (Math.exp(height)) * anchors[anchorInd][1];

                            out.add(new DetectedObject(k, centerX, centerY, width, height, classes_scores, prob));
                        }
                    }
                }
            }
        }
        return out;
    }
}
