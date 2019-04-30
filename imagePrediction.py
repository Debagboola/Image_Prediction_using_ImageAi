from imageai.Prediction import ImagePrediction
import os

execution_path = os.getcwd()

prediction = ImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(r"C:\Users\Agoola's\Desktop\computer vision projects\resnet50_weights_tf_dim_ordering_tf_kernels.h5")
prediction.loadModel()

predictions, probabilities = prediction.predictImage(r"C:\Users\Agoola's\Downloads\home-office-336373_1920 (1).jpg", result_count=10)

for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(eachPrediction, ":", eachProbability)