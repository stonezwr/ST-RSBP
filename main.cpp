#pragma warning (disable: 4819)
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <vector>
#include "net_spiking.cuh"
#include "common/cuMatrix.h"
#include "common/util.h"
#include "dataAugmentation/cuTransformation.cuh"
#include "common/Config.h"
#include "common/cuMatrixVector.h"
#include "common/Config.h"
#include "dataAugmentation/dataPretreatment.cuh"
#include "readData/readMnistData.h"
#include "readData/readNMnistData.h"
#include "readData/readSpeechData.h"
#include "readData/readNTidigits.h"
#include "common/track.h"


void runMnist();
void runNMnist();
void runFashionMnist();
void runTi46Alpha();
void runTi46Digits();
void runNTidigits();
bool init(cublasHandle_t& handle);

std::vector<int> g_argv;


int main (int argc, char** argv)
{
    //cudaDeviceReset();
    cudaSetDevice(0);
#ifdef linux
    signal(SIGSEGV, handler);   // install our handler
#endif
	srand(clock());
	if(argc >= 3){
		g_argv.push_back(atoi(argv[1]));
		g_argv.push_back(atoi(argv[2]));
	}
	printf("1. MNIST\n2. NMNIST\n3. Fashion-MNIST\n4. TI46_Alpha\n5. TI46_Digits\n6. N-Tidigits\n Choose the dataSet to run:");
	int cmd; 
	if(g_argv.size() >= 2)
		cmd = g_argv[0];
	else 
		if(1 != scanf("%d", &cmd)){
            LOG("scanf fail", "result/log.txt");
        }
    
    if(cmd == 1)
        runMnist();
	else if(cmd == 2)
        runNMnist();
    else if(cmd == 3)
        runFashionMnist();
    else if(cmd == 4)
        runTi46Alpha();
    else if(cmd == 5)
        runTi46Digits();
    else if(cmd == 6)
        runNTidigits();
	return EXIT_SUCCESS;
}

/*init cublas Handle*/
bool init(cublasHandle_t& handle)
{
	cublasStatus_t stat;
	stat = cublasCreate(&handle);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		printf ("init: CUBLAS initialization failed\n");
		exit(0);
	}
	return true;
}

void runMnist(){
	const int nclasses = 10;

 	//*state and cublas handle
 	cublasHandle_t handle;
	init(handle);
	
 	//* read the data from disk
	cuMatrixVector<bool> trainX;
	cuMatrixVector<bool> testX;
 	cuMatrix<int>* trainY, *testY;
    
    //* initialize the configuration
	Config * config = Config::instance();
    config->initPath("Config/SpikingMnistConfig.txt");
    ConfigDataSpiking * ds_config = (ConfigDataSpiking*)config->getLayerByName("data");
    int input_neurons = ds_config->m_inputNeurons;
    int end_time = config->getEndTime();
    int train_samples = config->getTrainSamples();
    int test_samples = config->getTestSamples();

 	readSpikingMnistData(trainX, trainY, "mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte", train_samples, input_neurons, end_time);
 	readSpikingMnistData(testX , testY, "mnist/t10k-images-idx3-ubyte", "mnist/t10k-labels-idx1-ubyte",  test_samples, input_neurons, end_time);

	MemoryMonitor::instance()->printCpuMemory();
	MemoryMonitor::instance()->printGpuMemory();

 	//* build SNN net
 	int ImgSize = 28; // fixed here for spiking mnist
	Config::instance()->setImageSize(ImgSize);
 	int nsamples = trainX.size();

 	int batch = Config::instance()->getBatchSize();
	float start,end;
    
	int cmd;
	cuInitDistortionMemery(batch, ImgSize);
    
	printf("1. random init weight\n2. Read weight from the checkpoint\nChoose the way to init weight:");

	if(g_argv.size() >= 2)
		cmd = g_argv[1];
	else 
		if(1 != scanf("%d", &cmd)){
            LOG("scanf fail", "result/log.txt");
        }
    
	buildSpikingNetwork(trainX.size(), testX.size());

    
	if(cmd == 2)
		cuReadSpikingNet("Result/checkPoint.txt");
    

	//* learning rate
	std::vector<float> nlrate;
	std::vector<float> nMomentum;
	std::vector<int> epoCount;

    int epochs = Config::instance()->getTestEpoch();
    for(int i = 0; i < epochs; ++i){
        nlrate.push_back(0.001f/sqrt(i+1)); nMomentum.push_back(0.90f);  epoCount.push_back(1);
    }

	start = clock();
	cuTrainSpikingNetwork(trainX, trainY, testX, testY, batch, nclasses, nlrate, nMomentum, epoCount, handle);
	end = clock();

	char logStr[1024];
	sprintf(logStr, "training time hours = %f\n", 
		(end - start) / CLOCKS_PER_SEC / 3600);
	LOG(logStr, "Result/log.txt");
}


void runNMnist(){
	const int nclasses = 10;

 	//*state and cublas handle
 	cublasHandle_t handle;
	init(handle);
	
 	//* read the data from disk
	cuMatrixVector<bool> trainX;
	cuMatrixVector<bool> testX;
 	cuMatrix<int>* trainY, *testY;
    
    //* initialize the configuration
	Config * config = Config::instance();
    config->initPath("Config/NMnistConfig.txt");

    ConfigDataSpiking * ds_config = (ConfigDataSpiking*)config->getLayerByName("data");
    int input_neurons = ds_config->m_inputNeurons;
    int end_time = config->getEndTime();
    int train_per_class = config->getTrainPerClass();
    int test_per_class = config->getTestPerClass();
 	readNMnistData(trainX, trainY, config->getTrainPath(), train_per_class, input_neurons, end_time);
 	readNMnistData(testX , testY, config->getTestPath(), test_per_class, input_neurons, end_time);

	MemoryMonitor::instance()->printCpuMemory();
	MemoryMonitor::instance()->printGpuMemory();

 	//* build SNN net 
 	int nsamples = trainX.size();

 	int batch = Config::instance()->getBatchSize();
	float start,end;
    
	int cmd;
	cuInitDistortionMemery(batch, 28);
    
	printf("1. random init weight\n2. Read weight from the checkpoint\nChoose the way to init weight:");

	if(g_argv.size() >= 2)
		cmd = g_argv[1];
	else 
		if(1 != scanf("%d", &cmd)){
            LOG("scanf fail", "result/log.txt");
        }
    
	buildSpikingNetwork(trainX.size(), testX.size());

    
	if(cmd == 2)
		cuReadSpikingNet("Result/checkPoint.txt");
    

	//* learning rate
	std::vector<float> nlrate;
	std::vector<float> nMomentum;
	std::vector<int> epoCount;

    int epochs = Config::instance()->getTestEpoch();
    for(int i = 0; i < epochs; ++i){
        nlrate.push_back(0.001f/sqrt(i+1)); nMomentum.push_back(0.90f);  epoCount.push_back(1);
    }

	start = clock();
	cuTrainSpikingNetwork(trainX, trainY, testX, testY, batch, nclasses, nlrate, nMomentum, epoCount, handle);
	end = clock();

	char logStr[1024];
	sprintf(logStr, "training time hours = %f\n", 
		(end - start) / CLOCKS_PER_SEC / 3600);
	LOG(logStr, "Result/log.txt");
}


void runFashionMnist(){
	const int nclasses = 10;

 	//*state and cublas handle
 	cublasHandle_t handle;
	init(handle);
	
 	//* read the data from disk
	cuMatrixVector<bool> trainX;
	cuMatrixVector<bool> testX;
 	cuMatrix<int>* trainY, *testY;
    
    //* initialize the configuration
	Config * config = Config::instance();
    config->initPath("Config/FashionMNIST.txt");

    ConfigDataSpiking * ds_config = (ConfigDataSpiking*)config->getLayerByName("data");
    int input_neurons = ds_config->m_inputNeurons;
    int end_time = config->getEndTime();
    int train_samples = config->getTrainSamples();
    int test_samples = config->getTestSamples();

 	readSpikingMnistData(trainX, trainY, "../../../fmnist/train-images-idx3-ubyte", "../../../fmnist/train-labels-idx1-ubyte", train_samples, input_neurons, end_time);
 	readSpikingMnistData(testX , testY, "../../../fmnist/t10k-images-idx3-ubyte", "../../../fmnist/t10k-labels-idx1-ubyte",  test_samples, input_neurons, end_time);

	MemoryMonitor::instance()->printCpuMemory();
	MemoryMonitor::instance()->printGpuMemory();

 	//* build SNN net
 	int ImgSize = 28; // fixed here for spiking mnist
	Config::instance()->setImageSize(ImgSize);
 	int nsamples = trainX.size();

 	int batch = Config::instance()->getBatchSize();
	float start,end;
    
	int cmd;
	cuInitDistortionMemery(batch, ImgSize);
    
	printf("1. random init weight\n2. Read weight from the checkpoint\nChoose the way to init weight:");

	if(g_argv.size() >= 2)
		cmd = g_argv[1];
	else 
		if(1 != scanf("%d", &cmd)){
            LOG("scanf fail", "result/log.txt");
        }
    
	buildSpikingNetwork(trainX.size(), testX.size());

    
	if(cmd == 2)
		cuReadSpikingNet("Result/checkPoint.txt");
    

	//* learning rate
	std::vector<float> nlrate;
	std::vector<float> nMomentum;
	std::vector<int> epoCount;

    int epochs = Config::instance()->getTestEpoch();
    for(int i = 0; i < epochs; ++i){
        nlrate.push_back(0.001f/sqrt(i+1)); nMomentum.push_back(0.90f);  epoCount.push_back(1);
    }

	start = clock();
	cuTrainSpikingNetwork(trainX, trainY, testX, testY, batch, nclasses, nlrate, nMomentum, epoCount, handle);
	end = clock();

	char logStr[1024];
	sprintf(logStr, "training time hours = %f\n", 
		(end - start) / CLOCKS_PER_SEC / 3600);
	LOG(logStr, "Result/log.txt");
}



void runTi46Alpha(){
	const int nclasses = 26;

 	//*state and cublas handle
 	cublasHandle_t handle;
	init(handle);
	
 	//* read the data from disk
	cuMatrixVector<bool> trainX;
	cuMatrixVector<bool> testX;
 	cuMatrix<int>* trainY, *testY;
    
    //* initialize the configuration
	Config * config = Config::instance();

    config->initPath("Config/Ti46_alpha.txt");
    ConfigDataSpiking * ds_config = (ConfigDataSpiking*)config->getLayerByName("data");
    int input_neurons = ds_config->m_inputNeurons;
    int end_time = config->getEndTime();
    int train_samples = config->getTrainSamples();
    int test_samples = config->getTestSamples();
 	readSpeechData(trainX, trainY, config->getTrainPath(), train_samples, input_neurons, end_time, nclasses);
 	readSpeechData(testX , testY,  config->getTestPath(), test_samples, input_neurons, end_time, nclasses);

	MemoryMonitor::instance()->printCpuMemory();
	MemoryMonitor::instance()->printGpuMemory();

 	//* build SNN net 
 	int nsamples = trainX.size();

 	int batch = Config::instance()->getBatchSize();
	float start,end;
    
	int cmd;
	printf("1. random init weight\n2. Read weight from the checkpoint\nChoose the way to init weight:");

	if(g_argv.size() >= 2)
		cmd = g_argv[1];
	else 
		if(1 != scanf("%d", &cmd)){
            LOG("scanf fail", "result/log.txt");
        }
    
	buildSpikingNetwork(trainX.size(), testX.size());

    
	if(cmd == 2)
	    cuReadSpikingNet("Result/checkPoint.txt");
    

	//* learning rate
	std::vector<float> nlrate;
	std::vector<float> nMomentum;
	std::vector<int> epoCount;
    int epochs = Config::instance()->getTestEpoch();

    for(int i = 0; i < epochs; ++i){
        nlrate.push_back(0.005f/sqrt(i+1)); nMomentum.push_back(0.90f);  epoCount.push_back(1);
    }

	start = clock();
	cuTrainSpikingNetwork(trainX, trainY, testX, testY, batch, nclasses, nlrate, nMomentum, epoCount, handle);
	end = clock();

	char logStr[1024];
	sprintf(logStr, "training time hours = %f\n", 
		(end - start) / CLOCKS_PER_SEC / 3600);
	LOG(logStr, "Result/log.txt");
}

void runTi46Digits(){
	const int nclasses = 10;

 	//*state and cublas handle
 	cublasHandle_t handle;
	init(handle);
	
 	//* read the data from disk
	cuMatrixVector<bool> trainX;
	cuMatrixVector<bool> testX;
 	cuMatrix<int>* trainY, *testY;
    
    //* initialize the configuration
	Config * config = Config::instance();
    config->initPath("Config/Ti46_digits.txt");
    ConfigDataSpiking * ds_config = (ConfigDataSpiking*)config->getLayerByName("data");
    int input_neurons = ds_config->m_inputNeurons;
    int end_time = config->getEndTime();
    int train_samples = config->getTrainSamples();
    int test_samples = config->getTestSamples();
 	readSpeechData(trainX, trainY, config->getTrainPath(), train_samples, input_neurons, end_time, nclasses);
 	readSpeechData(testX , testY,  config->getTestPath(), test_samples, input_neurons, end_time, nclasses);

	MemoryMonitor::instance()->printCpuMemory();
	MemoryMonitor::instance()->printGpuMemory();

 	//* build SNN net 
 	int nsamples = trainX.size();

 	int batch = Config::instance()->getBatchSize();
	float start,end;
    
	int cmd;
	printf("1. random init weight\n2. Read weight from the checkpoint\nChoose the way to init weight:");

	if(g_argv.size() >= 2)
		cmd = g_argv[1];
	else 
		if(1 != scanf("%d", &cmd)){
            LOG("scanf fail", "result/log.txt");
        }
    
	buildSpikingNetwork(trainX.size(), testX.size());

    
	if(cmd == 2)
	    cuReadSpikingNet("Result/checkPoint.txt");
    

	//* learning rate
	std::vector<float> nlrate;
	std::vector<float> nMomentum;
	std::vector<int> epoCount;

    int epochs = Config::instance()->getTestEpoch();
    for(int i = 0; i < epochs; ++i){
        nlrate.push_back(0.005f/sqrt(i+1)); nMomentum.push_back(0.90f);  epoCount.push_back(1);
    }

	start = clock();
	cuTrainSpikingNetwork(trainX, trainY, testX, testY, batch, nclasses, nlrate, nMomentum, epoCount, handle);
	end = clock();

	char logStr[1024];
	sprintf(logStr, "training time hours = %f\n", 
		(end - start) / CLOCKS_PER_SEC / 3600);
	LOG(logStr, "Result/log.txt");
}


void runNTidigits(){
	const int nclasses = 11;

 	//*state and cublas handle
 	cublasHandle_t handle;
	init(handle);
	
 	//* read the data from disk
	cuMatrixVector<bool> trainX;
	cuMatrixVector<bool> testX;
 	cuMatrix<int>* trainY, *testY;
    
    //* initialize the configuration
	Config * config = Config::instance();
    config->initPath("Config/N_TIDIGITS.txt");
    ConfigDataSpiking * ds_config = (ConfigDataSpiking*)config->getLayerByName("data");
    int input_neurons = ds_config->m_inputNeurons;
    int end_time = config->getEndTime();
    int train_samples = config->getTrainSamples();
    int test_samples = config->getTestSamples();
 	readNTidigits(trainX, trainY, config->getTrainPath(), train_samples, input_neurons, end_time, nclasses);
 	readNTidigits(testX , testY,  config->getTestPath(), test_samples, input_neurons, end_time, nclasses);

	MemoryMonitor::instance()->printCpuMemory();
	MemoryMonitor::instance()->printGpuMemory();

 	//* build SNN net 
 	int nsamples = trainX.size();

 	int batch = Config::instance()->getBatchSize();
	float start,end;
    
	int cmd;
	printf("1. random init weight\n2. Read weight from the checkpoint\nChoose the way to init weight:");

	if(g_argv.size() >= 2)
		cmd = g_argv[1];
	else 
		if(1 != scanf("%d", &cmd)){
            LOG("scanf fail", "result/log.txt");
        }
    
	buildSpikingNetwork(trainX.size(), testX.size());

    
	if(cmd == 2)
	    cuReadSpikingNet("Result/checkPoint.txt");
    

	//* learning rate
	std::vector<float> nlrate;
	std::vector<float> nMomentum;
	std::vector<int> epoCount;

    int epochs = Config::instance()->getTestEpoch();
    for(int i = 0; i < epochs; ++i){
        nlrate.push_back(0.005f/sqrt(i+1)); nMomentum.push_back(0.90f);  epoCount.push_back(1);
    }

	start = clock();
	cuTrainSpikingNetwork(trainX, trainY, testX, testY, batch, nclasses, nlrate, nMomentum, epoCount, handle);
	end = clock();

	char logStr[1024];
	sprintf(logStr, "training time hours = %f\n", 
		(end - start) / CLOCKS_PER_SEC / 3600);
	LOG(logStr, "Result/log.txt");
}

