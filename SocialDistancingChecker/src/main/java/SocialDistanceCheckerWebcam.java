import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
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
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.bytedeco.opencv.global.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.nd4j.common.util.MathUtils.sigmoid;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;
import static org.nd4j.linalg.ops.transforms.Transforms.euclideanDistance;

//  Author: Choo Wilson(yptheangel)
//  Special thanks to yquemener for writing yolov3 output interpreter
//  This example shows a Social Distancing Monitor running on your webcam
//  Set camera config on CameraStreamer.java
//  This code is still experimental and the performance is questionable.
//  Frame streaming happens on another thread while main thread reads a frame from the frame streaming thread and run inference
//  Due to the slow inference, the bounding box is not synced with the latest frame
// TODO: Try using FFMpegGrabber instead of OpenCV videocapture

public class SocialDistanceCheckerWebcam {

    private static CameraStreamer camStreamer = new CameraStreamer();
    private static ComputationGraph model;
    //        COCO has 80 classes
    private static int numClass = 80;
    private static int[][] anchors = {{10, 13}, {16, 30}, {33, 23}, {30, 61}, {62, 45}, {59, 119}, {116, 90}, {156, 198}, {373, 326}};
    private static int yolowidth = 416;
    private static int yoloheight = 416;
    private static INDArray input = null;
    private static Mat cvFrame = new Mat();


    public static void main(String[] args) throws InvalidKerasConfigurationException, IOException, UnsupportedKerasConfigurationException, InterruptedException {
        int safeDistance = 80;

        String modelPATH = "C:\\Users\\ChooWilson\\Downloads\\yolov3_416_fixed.h5";
        model = KerasModelImport.importKerasModelAndWeights(modelPATH);
        Thread modelInitThread = new Thread(() -> model.init());
        modelInitThread.start();

        // Start camera streamer thread
        camStreamer.startCapture();
        camStreamer.camStreamRunnable = () -> {
            try {
                cvFrame = camStreamer.getCVFrame();
            } catch (Exception e) {
                System.err.println("Error capturing frame. Error message: " + e.toString());
            }
        };

//        TODO:: Remove relying on this delay in order to make sure the camera stream is running before calling imshow
        Thread.sleep(1000);
        //TODO: Why is it when cvFrame is not empty OR VideoCap object is opened, the model init thread is already dead?
        while (!cvFrame.empty()) {
//            System.out.println(modelInitThread.isAlive());
            if (modelInitThread.isAlive()) {
                imshow("Social Distancing Monitor", cvFrame);
            } else {
                input = camStreamer.getNDArray(cvFrame);
                if (input != null) {
                    input = input.permute(0, 2, 3, 1);

                    List<DetectedObject> objs = getPredictedObjects(input);
                    YoloUtils.nms(objs, 0.4);

                    int w = cvFrame.cols();
                    int h = cvFrame.rows();
                    List<INDArray> centers = new ArrayList<>();
                    List<INDArray> people = new ArrayList<>();
                    Set violators = new HashSet<INDArray>();

                    int centerX;
                    int centerY;

                    for (DetectedObject obj : objs) {
                        if (obj.getPredictedClass() == 0) {
                            //            Scale the coordinates back to full size
                            centerX = (int) obj.getCenterX() * w / yolowidth;
                            centerY = (int) obj.getCenterY() * h / yoloheight;

                            circle(cvFrame, new Point(centerX, centerY), 2, new Scalar(0, 255, 0, 0), -1, 0, 0);
                            //            Draw bounding boxes on opencv mat
                            double[] xy1 = obj.getTopLeftXY();
                            double[] xy2 = obj.getBottomRightXY();
                            //            Scale the coordinates back to full size
                            xy1[0] = xy1[0] * w / yolowidth;
                            xy1[1] = xy1[1] * h / yoloheight;
                            xy2[0] = xy2[0] * w / yolowidth;
                            xy2[1] = xy2[1] * h / yoloheight;

                            //Draw bounding box
                            rectangle(cvFrame, new Point((int) xy1[0], (int) xy1[1]), new Point((int) xy2[0], (int) xy2[1]), new Scalar(0, 255, 0, 0), 2, LINE_8, 0);
                            centers.add(Nd4j.create(new float[]{(float) centerX, (float) centerY}));
                            people.add(Nd4j.create(new float[]{(float) xy1[0], (float) xy1[1], (float) xy2[0], (float) xy2[1]}));
                        }
                    }
                    //        Calculate the euclidean distance between all pairs of center points
                    for (int i = 0; i < centers.size(); i++) {
                        for (int j = 0; j < centers.size(); j++) {
                            double distance = euclideanDistance(centers.get(i), centers.get(j));
                            if (distance < safeDistance && distance > 0) {
                                line(cvFrame, new Point(centers.get(i).getInt(0), centers.get(i).getInt(1)),
                                        new Point(centers.get(j).getInt(0), centers.get(j).getInt(1)), Scalar.RED, 2, 1, 0);

                                violators.add(centers.get(i));
                                violators.add(centers.get(j));

                                int xmin = people.get(i).getInt(0);
                                int ymin = people.get(i).getInt(1);
                                int xmax = people.get(i).getInt(2);
                                int ymax = people.get(i).getInt(3);

                                rectangle(cvFrame, new Point(xmin, ymin), new Point(xmax, ymax), Scalar.RED, 2, LINE_8, 0);
                                circle(cvFrame, new Point(centers.get(i).getInt(0), centers.get(i).getInt(1)), 3, Scalar.RED, -1, 0, 0);
                            }
                        }
                    }
                    putText(cvFrame, String.format("Number of people: %d", people.size()), new Point(10, 30), 4, 1.0, new Scalar(0, 255, 0, 0), 2, LINE_8, false);
                    putText(cvFrame, String.format("Number of violators: %d", violators.size()), new Point(10, 60), 4, 1.0, new Scalar(0, 0, 255, 0), 2, LINE_8, false);
                }
                imshow("Social Distancing Monitor", cvFrame);
            }
//            imshow("Social Distancing Monitor", cvFrame);

            //    Press Esc key to quit
            if (waitKey(33) == 27) {
                destroyAllWindows();
                break;
            }
        }
    }

    public static List<DetectedObject> getPredictedObjects(INDArray input) {

        INDArray[] outputs = model.output(input);
        List<DetectedObject> out = new ArrayList();
        float detectionThreshold = 0.6f;
        // Each cell had information for 3 boxes
        int[] boxOffsets = {0, numClass + 5, (numClass + 5) * 2};
        int exampleNum_in_batch = 0;

        for (int layerNum = 0; layerNum < 3; layerNum++) {
            long gridWidth = outputs[layerNum].shape()[1];
            long gridHeight = outputs[layerNum].shape()[2];
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
                        float prob = outputs[layerNum].getFloat(new int[]{exampleNum_in_batch, i, j, boxOffsets[k] + 4});
                        if (prob > detectionThreshold) {
//                            TODO: class probabilities does not make sense
                            INDArray classes_scores = outputs[layerNum].get(
                                    point(exampleNum_in_batch),
                                    point(i),
                                    point(j),
                                    NDArrayIndex.interval(boxOffsets[k] + 5, boxOffsets[k] + numClass + 5));

                            centerX = outputs[layerNum].getFloat(new int[]{exampleNum_in_batch, i, j, boxOffsets[k] + 0});
                            centerY = outputs[layerNum].getFloat(new int[]{exampleNum_in_batch, i, j, boxOffsets[k] + 1});
                            width = outputs[layerNum].getFloat(new int[]{exampleNum_in_batch, i, j, boxOffsets[k] + 2});
                            height = outputs[layerNum].getFloat(new int[]{exampleNum_in_batch, i, j, boxOffsets[k] + 3});
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
