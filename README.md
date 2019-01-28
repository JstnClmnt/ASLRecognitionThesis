# ASLRecognitionThesis

American Sign Language (ASL) Recognition is a prominent topic in the field of Computer Vision and Artificial Intelligence. In recent years, Dynamic Time Warping (DTW), Hidden Markov Models (HMM), Neural Networks are used together with Microsoft Kinect to recognize American sign language gestures.
In this paper, a new approach to American Sign Language is created. By using a web camera or a mobile camera with 91 gestures, each composed of fingerspelling and dynamic gestures. The research aims to surpass the environmental limitations of webcam-related ASL recognition, which only shows the hand of the given signer.
By using Pose-Estimation with Part Affinity Fields for feature extraction and pre-processing of the given input, such as video to overcome the environmental limitations, and Hidden Markov Models for the recognition of a gesture, the research has obtained an accuracy of 64% for fingerspelling gestures, 43% for dynamic gestures, and 53.5% when the previous two cases are combined.
A reason for the low accuracy can be because of the information loss of the input. Since this research is only limited to 10-12 frames per input, if the input has 60 frames, only 10-12 frames are then obtained and processed to be recognized. However, despite the information loss, this research yields an accuracy above average and has broken the environmental limitations of previous ASL Recognition researches using Web Camera.
Albeit falling behind in terms of accuracy compared to researches that used Microsoft Kinect or any similar camera devices augmented with depth sensors, this research shows a possibility of using a robust feature extraction library with a time-series classification algorithm like the Hidden Markov Model to be able to replicate the performance of gesture systems that utilize Kinect.

*Will provide an guide on running the code soon. Code is still under enhancement despite the research being completed.*

<br> [Full Text](https://drive.google.com/file/d/1m68deZkKmpZxtF4cDiZ7-0LOMHadRRkv/view?usp=sharing)</br>
<br> [Summary](https://drive.google.com/file/d/1mGsWyrl3OpE33FUF6Bw2ukQ90D6_he_D/view?usp=sharing)</br>
