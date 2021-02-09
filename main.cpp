#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

using namespace std;
using namespace cv;
using namespace cv::face;

float getDistance(const Point2f& p1, const Point2f& p2);
string getEmotion(InputArray frame, InputArray lastFrame, CascadeClassifier classifier,
                  vector< vector <Point2f> > landmarks, const vector<Rect>& faces, const Ptr<Facemark>& facemark);

int main(int argc, char** argv){
    string label;
    bool firstFrame = true;
    CascadeClassifier faceDetector;
    Mat frame, gray, lastFrame;

    vector<Rect> faces;

    vector< vector<Point2f> > landmarks;

    // First, we verify that we have the correct number of arguments and we parse them
    if(argc < 3){
        cerr << "The correct use has the form of [path to cascade classifier] [path to model for facemark detection]";
    }

    string cascadeFileName = argv[1];

    string modelFileName = argv[2];



    // The cascade classifier is loaded from the given file
    try{
        faceDetector.load(cascadeFileName);
    } catch (int e) {
        cerr << e;
        cerr << "There was an error opening your cascade classifier, please check your inputs and try again.";
        return 1;
    }
    // We declare the LBF facemark detector
    Ptr<Facemark> facemark = FacemarkLBF::create();

    // The model is loaded
    facemark->loadModel(modelFileName);

    // The video capture object is loaded as a webcam
    VideoCapture webCam(0);


    // While the webcams captures something
    while (webCam.read(frame)){
        // We clear the faces and landmarks matrices
        faces.clear();
        landmarks.clear();

        // Convert the captured frame to gray scale and is stored in the gray matrix
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        //If this is the first frame, we copy the current frame as the last one
        if(firstFrame){
            lastFrame = gray.clone();
            firstFrame = false;
        }
        // The faces are detected and stored in the faces vector
        faceDetector.detectMultiScale(gray, faces);

        // We try to find the points for the landmarks
        bool success = facemark->fit(frame, faces, landmarks);

        // If it is successful, we draw the points for all the faces detected
        if(success){
            for(auto & landmark : landmarks){
                drawFacemarks(frame, landmark, Scalar(255, 0, 0));
            }
            // With the current frame analyzed, we compare it to the previous frame
            label = getEmotion(gray, lastFrame, faceDetector, landmarks, faces, facemark);
        }else{
            cout << "Could not detect facemarks :(" << endl;
            label = "No faces detected";
        }




        // The label is inserted in the frame
        putText(frame, label, Point(5, 200), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0));

        // The image is shown
        imshow("Facial Landmark Detection", frame);

        // Once the process is complete, we store the current frame and start the cycle again
        lastFrame = gray.clone();

        // If the user presses EXIT, the program closes
        if (waitKey(1) == 27) break;
    }
    return 0;
}

// Function dedicated to obtain the basic emotion according to Paul Ekman's theory on micro-expressions. It receives
// both the current frame and last frame, a cascade classifier, a couple vectors containing both the lanndmarks (points)
// and faces (rectangles) and an initialized facemark pointer (in this case, an LBF one)
string getEmotion(InputArray frame, InputArray lastFrame, CascadeClassifier classifier,
                  vector< vector <Point2f> > landmarks, const vector<Rect>& faces, const Ptr<Facemark>& facemark){
    string result;
    vector< vector <Point2f> > lastLandmarks;
    vector<Rect> lastFaces;

    classifier.detectMultiScale(lastFrame, lastFaces);
    facemark->fit(lastFrame, lastFaces, lastLandmarks);


    // AU 1
    float dist21to33 = getDistance(landmarks[0][21], landmarks[0][33]) -
            getDistance(lastLandmarks[0][21], lastLandmarks[0][33]);
    float dist22to33 = getDistance(landmarks[0][22], landmarks[0][33]) -
            getDistance(lastLandmarks[0][22], lastLandmarks[0][33]);
    float dist20to33 = getDistance(landmarks[0][20], landmarks[0][33]) -
            getDistance(lastLandmarks[0][20], lastLandmarks[0][33]);
    float dist23to33 = getDistance(landmarks[0][27], landmarks[0][33]) -
            getDistance(lastLandmarks[0][23], lastLandmarks[0][33]);

    // AU 2
    float dist17to36 = getDistance(landmarks[0][17], landmarks[0][36]) -
            getDistance(lastLandmarks[0][17], lastLandmarks[0][36]);
    float dist26to45 = getDistance(landmarks[0][17], landmarks[0][36]) -
            getDistance(lastLandmarks[0][17], lastLandmarks[0][36]);

    // AU 4
    float dist21to22 = getDistance(landmarks[0][21], landmarks[0][22]) -
            getDistance(lastLandmarks[0][21], lastLandmarks[0][22]);
    float dist20to23 = getDistance(landmarks[0][20], landmarks[0][23]) -
            getDistance(lastLandmarks[0][20], lastLandmarks[0][23]);

    // AU 5
    float dist33to38 = getDistance(landmarks[0][33], landmarks[0][38]) -
            getDistance(lastLandmarks[0][33], lastLandmarks[0][38]);
    float dist33to37 = getDistance(landmarks[0][33], landmarks[0][37]) -
            getDistance(lastLandmarks[0][33], lastLandmarks[0][37]);
    float dist33to43 = getDistance(landmarks[0][33], landmarks[0][43]) -
            getDistance(lastLandmarks[0][33], lastLandmarks[0][43]);
    float dist33to44 = getDistance(landmarks[0][44], landmarks[0][44]) -
            getDistance(lastLandmarks[0][33], lastLandmarks[0][44]);

    // AU 7
    float dist27to40 = getDistance(landmarks[0][27], landmarks[0][40]) -
            getDistance(lastLandmarks[0][27], lastLandmarks[0][40]);
    float dist27to47 = getDistance(landmarks[0][27], landmarks[0][47]) -
            getDistance(lastLandmarks[0][27], lastLandmarks[0][47]);

    // AU 9 = AU4 + AU1

    // AU 12
    float dist36to48 = getDistance(landmarks[0][36], landmarks[0][48]) -
            getDistance(lastLandmarks[0][36], lastLandmarks[0][48]);
    float dist45to54 = getDistance(landmarks[0][45], landmarks[0][54]) -
            getDistance(lastLandmarks[0][45], lastLandmarks[0][54]);

    // AU 15
    float dist5to48 = getDistance(landmarks[0][5], landmarks[0][48]) -
            getDistance(lastLandmarks[0][5], lastLandmarks[0][48]);
    float dist11to54= getDistance(landmarks[0][11], landmarks[0][54]) -
            getDistance(lastLandmarks[0][11], lastLandmarks[0][54]);

    // AU 16
    float dist33to57 = getDistance(landmarks[0][33], landmarks[0][57]) -
            getDistance(lastLandmarks[0][33], lastLandmarks[0][57]);
    float dist33to56 = getDistance(landmarks[0][33], landmarks[0][56]) -
            getDistance(lastLandmarks[0][33], lastLandmarks[0][56]);
    float dist33to58 = getDistance(landmarks[0][33], landmarks[0][58]) -
            getDistance(lastLandmarks[0][33], lastLandmarks[0][58]);

    // AU 17 -> 8to33

    // AU 23 -> AU25 & AU26

    // AU 24
    float dist50to61 = getDistance(landmarks[0][50], landmarks[0][61]) -
            getDistance(lastLandmarks[0][50], lastLandmarks[0][61]);
    float dist51to62 = getDistance(landmarks[0][51], landmarks[0][62]) -
            getDistance(lastLandmarks[0][51], lastLandmarks[0][62]);
    float dist52to63 = getDistance(landmarks[0][52], landmarks[0][63]) -
            getDistance(lastLandmarks[0][52], lastLandmarks[0][63]);

    // AU 25 & 26
    float dist61to67 = getDistance(landmarks[0][61], landmarks[0][67]) -
            getDistance(lastLandmarks[0][61], lastLandmarks[0][67]);
    float dist62to66 = getDistance(landmarks[0][62], landmarks[0][66]) -
            getDistance(lastLandmarks[0][62], lastLandmarks[0][66]);
    float dist63to65 = getDistance(landmarks[0][63], landmarks[0][65]) -
            getDistance(lastLandmarks[0][63], lastLandmarks[0][65]);

    // AU 27
    float dist8to33 = getDistance(landmarks[0][8], landmarks[0][33]) -
            getDistance(lastLandmarks[0][8], lastLandmarks[0][33]);
    float dist7to33 = getDistance(landmarks[0][7], landmarks[0][33]) -
            getDistance(lastLandmarks[0][7], lastLandmarks[0][33]);
    float dist9to33 = getDistance(landmarks[0][9], landmarks[0][33]) -
            getDistance(lastLandmarks[0][9], lastLandmarks[0][33]);


    if(dist21to33 > 0 && dist22to33 > 0 && dist20to33 > 0 && dist23to33 > 0 && // AU 1
       dist17to36 > 0 && dist26to45 > 0 && // AU 2
       dist33to38 > 0 && dist33to37 > 0 && dist33to43 > 0 && dist33to44 > 0  && // AU 5
       (dist61to67 > 0 && dist62to66 > 0 && dist63to65 > 0 && // AU 26
        dist8to33 > 0 && dist7to33 > 0 && dist9to33 > 0) // AU 27
       )
        result = "Surprise";
    else if(dist21to33 > 0 && dist22to33 > 0 && dist20to33 > 0 && dist23to33 > 0 && // AU 1
        dist21to22 < 0 && dist20to23 < 0 && // AU 4
        dist5to48 < 0 && dist11to54 < 0) // AU 15
        result = "Sadness";
    else if(dist21to33 > 0 && dist22to33 > 0 && dist20to33 > 0 && dist23to33 > 0 && // AU 1
        dist17to36 > 0 && dist26to45 > 0 && // AU 2
        dist21to22 < 0 && dist20to23 < 0 && // AU 4
        dist33to37 > 0 && dist33to38 > 0 && dist33to43 > 0 && dist33to44 > 0 && // AU 5
        (dist61to67 > 0 && dist62to66 > 0 && dist63to65 > 0 && // AU 26
         dist8to33 > 0 && dist7to33 > 0 && dist9to33 > 0) // AU 27
        )
        result = "Fear";
    else if(dist21to22 < 0 && dist20to23 < 0 && dist21to33 < 0 && dist22to33 < 0 && // AU 9
        dist33to56 > 0 && dist33to57 > 0 && dist33to58 > 0 && // AU 16
        ((dist5to48 < 0 && dist11to54 < 0) || (dist61to67 > 0 && dist62to66 > 0 && dist63to65 > 0) ) // AU 15 or AU 26
        )
        result = "Disgust";
    else if(dist21to22 < 0 && dist20to23 < 0 && // AU 4
            dist33to38 > 0 && dist33to37 > 0 && dist33to43 > 0 && dist33to44 > 0 &&// AU 5
            dist27to40 < 0 && dist27to47 < 0 && // AU 7
            ((dist61to67 < 0 && dist62to66 < 0 && dist63to65 < 0 ) || // AU 23 or AU 24
            (dist50to61 < 0 && dist51to62 < 0 && dist52to63 < 0)))
        result = "Angry";
    else if(dist36to48 < 0 && dist45to54 < 0) // AU 12
        result = "Happiness";
    else
        result = "No emotion detected";

    return result;
}

float getDistance(const Point2f& p1, const Point2f& p2){
    return sqrt(powf((p1.x - p2.x),2) + powf((p1.y - p2.y),2));
}