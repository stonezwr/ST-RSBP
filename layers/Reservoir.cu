#include "Reservoir.h"
#include "../common/cuBase.h"
#include "../common/Config.h"
#include "../common/util.h"
#include "../readData/readSpeechData.h"
#include <fstream>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusolverDn.h"
#include "helper_cuda.h"

#include "helper_cusolver.h"


__global__ void g_Reservoir_feedforward(
    float* inputs_resp,
	float* w,
	float* w_l,
    float* b,
	bool*  outputs,
    int* fireCount,
	int* connection,
	int inputSize,
	int outputSize,
    int endTime,
    float vth,
    int dummyFreq, 
    int T_REFRAC,
    float TAU_M,
    float TAU_S);

__device__ float d_Reservoir_accumulate_spikes(
    int inputSize,
    int outputSize,
    float* input_response,
    bool* output,
    int o_idx,
    float* weights,
    float* weights_lat,
    float* biases,
    int t,
    int dummyFreq,
    int endTime,
	int* connection);


__global__ void g_Reservoir_synaptic_effect(
        int* inputs_time,
        int* outputs_time,
        int* batchPreFireCount,
        int* batchFireCount,
        float* batchAccEffect,
        int inputSize,
        int outputSize,
        int endTime,
        int T_REFRAC,
        float TAU_M,
        float TAU_S);


__global__ void g_Reservoir_synaptic_effect_reservoir(
		float* w_l,
        int* outputs_time,
        int* batchFireCount,
        float* reservoirEffect,
        int outputSize,
        int endTime,
        int T_REFRAC,
        float TAU_M,
        float TAU_S);


__global__ void g_Reservoir_sum_effect_ratio_input(
        float* weights,
		float* sumEffectInput,
        int* batchPreFireCount,
        int* batchFireCount,
		float* effectPoly,
		int out_size,
		int degree,
        int inputSize,
		int outputSize,
		float vth);

__global__ void g_Reservoir_sum_effect_ratio_reservoir(
        float* w_l,
		float* sumEffectReservoir,
        int* batchFireCount,
		float* effectPoly,
		int out_size,
		int degree,
		int outputSize,
		float vth);

__global__ void g_Reservoir_effect_ratio_LHS(
		float* w_l,
        int* batchFireCount,
        float* reservoirEffect,
        float* m_LHS,
		float* sumEffectReservoir,
		float* sumEffectInput,
        int outputSize,
		float vth);

__global__ void g_Reservoir_effect_ratio_RHS(
		float* weights,
        int* batchPreFireCount,
        int* batchFireCount,
        float* batchAccEffect,
        float* m_RHS,
		int inputSize,
        int outputSize,
		float vth);

__global__ void g_Reservoir_copy_to_vector(
		float* from,
		float* to,
		int i_idx,
		int outputSize);

__global__ void g_Reservoir_copy_from_vector(
		float* to,
		float* from,
		int i_idx,
		int outputSize);

__global__ void g_Reservoir_wgrad_spiketime(
        float* batchAccEffect,
        float* curDelta,
        float* wgradTmp,
		float* sumEffectReservoir,
		float* sumEffectInput,
        int inputSize,
        int outputSize);

__global__ void g_Reservoir_wgrad_spiketime_reservoir(
		float* w_l,
        float* reservoirEffect,
        float* curDelta,
        float* wgradTmp_reservoir,
		float* sumEffectReservoir,
		float* sumEffectInput,
        int outputSize);

__global__ void g_Reservoir_setSquareSum(
    float* w_sq_sum,
    int outputSize,
	float v);

__global__ void g_Reservoir_calSquareSum(
    float* w,
    float* w_sq_sum,
    int outputSize,
    int inputSize,
    float weight_limit);

__global__ void g_Reservoir_calSquareSum_reservoir(
    float* w_l,
    float* w_sq_sum,
    int outputSize,
    float weight_limit);

__global__ void g_Reservoir_divideSquareSum(
    float* w_sq_sum,
    int outputSize,
	int count);

__global__ void g_Reservoir_gradAdd(
	float* wgradTmp,
	float* wgrad,
	float* w,
    float* w_sq_sum,
	float lambda,
    float beta,
    float limit,
    int inputSize,
	int outputSize);

__global__ void g_Reservoir_gradAdd_reservoir(
	float* wgradTmp_reservoir,
	float* wgrad_reservor,
	float* w_l,
    float* w_sq_sum,
	float lambda,
    float beta,
    float limit,
	int outputSize);

__device__ float d_Reservoir_Compute_poly(
   int i_cnt, 
   int degree, 
   float *effectPoly, 
   int o_cnt);

void Reservoir::feedforward()
{
    if((inputs == NULL))
    {
        printf("Reservoir init error\n");
        exit(0);
    }

    // fast input response
    g_cast_bool_2_float<<<dim3(batch, endTime), min(1024, inputSize)>>>(inputs->getDev(), endTime, inputSize, inputs->cols, inputs->channels, batch, inputs_float->getDev());
    matrixMul(w, inputs_float, inputs_resp_tmp); //input_resp_tmp rows:outputSize; cols:endTime*batch
    g_transform_2_batch<<<dim3(batch, outputSize), min(1024, endTime)>>>(inputs_resp_tmp->getDev(), endTime, outputSize, batch, inputs_resp->getDev());

    // convert (batch, inputDim2*endTime, amount) to (batch, amount*inputDim2*endTime, 1)
    g_convert_spiketimes<<<dim3(batch, endTime), min(1024, inputSize)>>>(inputs_time->getDev(), endTime, inputSize, inputs_time->cols, inputs_time->channels, batch, inputs_time_format->getDev());
    // convert (batch, inputDim2, amount) to (batch, amount*inputDim2, 1)
    g_convert_firecounts<<<dim3(batch), min(1024, inputSize)>>>(preFireCount->getDev(), preFireCount->getArea(), inputSize, preFireCount->cols, preFireCount->channels, batch, preFireCount_format->getDev());

    dim3 thread= dim3(min(1024, outputSize));
    dim3 block = dim3(batch);
    ConfigReservoir * config = (ConfigReservoir*) Config::instance()->getLayerByName(m_name); 
    int dummyFreq = config->getBiasFreq();

    
	g_Reservoir_feedforward<<<block, thread>>>(
        inputs_resp->getDev(),
        w->getDev(),
        w_laterial->getDev(),
        b->getDev(),
        outputs->getDev(),
        fireCount->getDev(),
		reservoir_connection->getDev(),
        inputSize,
        outputSize,
        endTime,
        threshold,
        dummyFreq,
        T_REFRAC,
        TAU_M,
        TAU_S);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("Reservoir::g_Reservoir_feedforward");

    block = dim3(batch, 1);
    thread = dim3(min(outputSize, 1024));

    // transform the binary response matrix to the spike times
    g_response_2_spiketime<<<block, thread>>>(
            outputs->getDev(),
            outputs_time->getDev(),
            outputs->getArea(),
            outputSize,
            endTime);
    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("Reservoir:g_response_2_spiketime");

}


void Reservoir::backpropagation()
{ 
    // compute the accumulative synaptic effect of input synapses
    dim3 thread = dim3(min(1024, outputSize));
    dim3 block  = dim3(batch, inputSize);
    cudaFuncSetCacheConfig(g_Reservoir_synaptic_effect, cudaFuncCachePreferL1);
    g_Reservoir_synaptic_effect<<<block, thread>>>(
        inputs_time_format->getDev(),
        outputs_time->getDev(),
        preFireCount_format->getDev(),
        fireCount->getDev(),
        accEffect->getDev(),
        inputSize,
        outputSize,
        endTime,
        T_REFRAC,
        TAU_M,
        TAU_S);
    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("g_Reservoir_synaptic_effect");
   
    // compute the accumulative synaptic effect of reservoir synapses
    thread = dim3(min(1024, outputSize));
    block  = dim3(batch, outputSize);
    cudaFuncSetCacheConfig(g_Reservoir_synaptic_effect_reservoir, cudaFuncCachePreferL1);
    g_Reservoir_synaptic_effect_reservoir<<<block, thread>>>(
		w_laterial->getDev(),
        outputs_time->getDev(),
        fireCount->getDev(),
        reservoirEffect->getDev(),
        outputSize,
        endTime,
        T_REFRAC,
        TAU_M,
        TAU_S);
    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("g_Reservoir_synaptic_effect_reservoir");

	thread = dim3(min(1024, inputSize));
	block  = dim3(batch, outputSize);
	g_Reservoir_sum_effect_ratio_input<<<block, thread, sizeof(float) * min(1024, inputSize)>>>(
		w->getDev(),
		sumEffectRatioInput->getDev(),
        preFireCount_format->getDev(),
    	fireCount->getDev(),
		effectPoly->getDev(),
		50,
		5,
		inputSize,
		outputSize,
		threshold);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("g_Reservoir_sum_effect_ratio_input");

	thread = dim3(min(1024, outputSize));
	block  = dim3(batch, outputSize);
	g_Reservoir_sum_effect_ratio_reservoir<<<block, thread, sizeof(float) * min(1024, outputSize)>>>(
		w_laterial->getDev(),
		sumEffectRatioReservoir->getDev(),
    	fireCount->getDev(),
		effectPoly->getDev(),
		50,
		5,
		outputSize,
		threshold);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("g_Reservoir_sum_effect_ratio_reservoir");


    // divide the curDelta by vth
    block = dim3(batch, 1);
    thread = dim3(min(1024, outputSize));
    g_divide_by_threshold<<<block, thread>>>(curDelta->getDev(), curDelta->getArea(), curDelta->cols, threshold);
    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("g_divide_by_threshold");
    
    // compute preDelta: curDelta: batch * outputSize; w: outputSize * inputSize
    if(preDelta == NULL){
        ConfigSpiking* config = (ConfigSpiking*)Config::instance()->getLayerByName(m_name);
        assert(config->m_input == "data");
    }
    else{
		thread = dim3(min(1024, outputSize));
		block  = dim3(batch, outputSize);
		g_Reservoir_effect_ratio_LHS<<<block, thread>>>(
			w_laterial->getDev(),
			fireCount->getDev(),
			reservoirEffect->getDev(),
			matrixLHS->getDev(),
			sumEffectRatioReservoir->getDev(),
			sumEffectRatioInput->getDev(),
			outputSize,
			threshold);
		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("g_Reservoir_effect_ratio_LHS");

		thread = dim3(min(1024, inputSize));
		block  = dim3(batch, outputSize);
		g_Reservoir_effect_ratio_RHS<<<block, thread>>>(
			w->getDev(),
	        preFireCount_format->getDev(),
	        fireCount->getDev(),
			accEffect->getDev(),
			matrixRHS->getDev(),
			inputSize,
			outputSize,
			threshold);
		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("g_Reservoir_effect_ratio_RHS");
	
		linearSolverQR(
			matrixLHS,
			matrixRHS,
			effectRatio,
			outputSize,
			outputSize,
			inputSize);

		matrixMul(curDelta, effectRatio, preDelta_format);
		block = batch;
		thread = min(512, preDelta->channels * preDelta->cols);
		g_preDeltaFormat<<<block, thread>>>(preDelta_format->getDev(), preDelta->getDev(),
		preDelta->rows, preDelta->cols, preDelta->channels);
		cudaStreamSynchronize(0);
		getLastCudaError("g_preDeltaFormat");
	}
}


void Reservoir::getGrad()
{
    dim3 thread = dim3(min(1024, inputSize));
    dim3 block  = dim3(batch, outputSize);
   
    cudaFuncSetCacheConfig(g_Reservoir_wgrad_spiketime,cudaFuncCachePreferL1);

    g_Reservoir_wgrad_spiketime<<<block, thread>>>(
        accEffect->getDev(),
        curDelta->getDev(),
        wgradTmp->getDev(),
		sumEffectRatioReservoir->getDev(),
		sumEffectRatioInput->getDev(),
        inputSize,
        outputSize);

    checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("g_Reservoir_wgrad_spiketime");

	if(wgradTmp_reservoir!=NULL){
		thread = dim3(min(1024, outputSize));
		block  = dim3(batch, outputSize);
   
		cudaFuncSetCacheConfig(g_Reservoir_wgrad_spiketime_reservoir,cudaFuncCachePreferL1);

		g_Reservoir_wgrad_spiketime_reservoir<<<block, thread>>>(
			w_laterial->getDev(),
			reservoirEffect->getDev(),
			curDelta->getDev(),
			wgradTmp_reservoir->getDev(),
			sumEffectRatioReservoir->getDev(),
			sumEffectRatioInput->getDev(),
			outputSize);

		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("g_Reservoir_wgrad_spiketime_reservoir");
	}

    
    block = dim3(1);
    thread = dim3(min(outputSize, 1024));
    g_Reservoir_setSquareSum<<<block, thread>>>(
        weightSqSum->getDev(),
        outputSize,
		0);
    checkCudaErrors(cudaStreamSynchronize(0));    
	getLastCudaError("g_Reservoir_setSquareSum");

    block = dim3(outputSize);
    thread = dim3(min(inputSize, 1024));

    g_Reservoir_calSquareSum<<<block, thread, sizeof(float) * min(inputSize, 1024)>>>(
        w->getDev(),
        weightSqSum->getDev(),
        outputSize,
        inputSize,
        weightLimit);
    checkCudaErrors(cudaStreamSynchronize(0));    
	getLastCudaError("g_Reservoir_calSquareSum");
 
	if(wgradTmp_reservoir!=NULL){
		block = dim3(outputSize);
		thread = dim3(min(outputSize, 1024));
		g_Reservoir_calSquareSum_reservoir<<<block, thread, sizeof(float) * min(outputSize, 1024)>>>(
			w_laterial->getDev(),
			weightSqSum->getDev(),
			outputSize,
			weightLimit);
		checkCudaErrors(cudaStreamSynchronize(0));    
		getLastCudaError("g_Reservoir_calSquareSum_reservoir");
		
		block = dim3(1);
		thread = dim3(min(outputSize, 1024));
		g_Reservoir_divideSquareSum<<<block, thread>>>(
			weightSqSum->getDev(),
			outputSize,
			reservoirSize+inputSize);
		checkCudaErrors(cudaStreamSynchronize(0));    
		getLastCudaError("g_Reservoir_setSquareSum");
	}else{
		block = dim3(1);
		thread = dim3(min(outputSize, 1024));
		g_Reservoir_divideSquareSum<<<block, thread>>>(
			weightSqSum->getDev(),
			outputSize,
			inputSize);
		checkCudaErrors(cudaStreamSynchronize(0));    
		getLastCudaError("g_Reservoir_setSquareSum");
	}



	block  = dim3(outputSize);
    thread = dim3(min(inputSize, 1024));

	g_Reservoir_gradAdd<<<block, thread>>>(
		wgradTmp->getDev(),
		wgrad->getDev(),
		w->getDev(),
        weightSqSum->getDev(),
		lambda,
        beta,
        weightLimit,
        inputSize,
		outputSize);

	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("g_Reservoir_gradAdd");



	if(wgradTmp_reservoir!=NULL){
		block  = dim3(outputSize);
    	thread = dim3(min(outputSize, 1024));

		g_Reservoir_gradAdd_reservoir<<<block, thread>>>(
			wgradTmp_reservoir->getDev(),
			wgrad_reservoir->getDev(),
			w_laterial->getDev(),
    	    weightSqSum->getDev(),
			lambda,
    	    beta,
    	    weightLimit,
			outputSize);

		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("g_Reservoir_gradAdd_reservoir");
	}
    
}	

void Reservoir::updateWeight()
{
    dim3 block  = min((w->getLen() + 255)/ 256, 5120);
    dim3 thread = 256;

    assert(Config::instance()->getOptimizerType() == std::string("adam"));
    g_adam_vecAdd<<<block, thread, 0, Layers::instance()->get_stream()>>>(
        g1_w->getDev(),
        g2_w->getDev(),
        b1_t,
        b2_t,
        wgrad->getDev(),
        w->getDev(),
        w->getLen(),
        Config::instance()->getLrate());
	if(wgradTmp_reservoir!=NULL){
		g_adam_vecAdd_reservoir<<<block, thread, 0, Layers::instance()->get_stream()>>>(
			g1_w_reservoir->getDev(),
            g2_w_reservoir->getDev(),
            b1_t,
            b2_t,
            wgrad_reservoir->getDev(),
            w_laterial->getDev(),
            w_laterial->getLen(),
            Config::instance()->getLrate());
	}
    b1_t *= 0.9f; b2_t *= 0.999f;
}


Reservoir::Reservoir(std::string name)
{
	m_name = name;
	ConfigReservoir* config = (ConfigReservoir*)Config::instance()->getLayerByName(m_name);
	SpikingLayerBase * preLayer = (SpikingLayerBase*)Layers::instance()->get(config->m_input);

	inputs = preLayer->getSpikingOutputs();
    inputs_time = preLayer->getSpikingTimeOutputs();
    inputs_time_format = new cuMatrix<int>(inputs_time->rows, inputs_time->cols * inputs_time->channels, 1);
	preDelta = preLayer->getCurDelta();
	std::cout<<preLayer->m_name<<std::endl;
	preDelta_format = NULL;
    if(preDelta != NULL){
		preDelta_format = new cuMatrix<float>(preDelta->rows, preDelta->cols * preDelta->channels, 1);
	}

    preFireCount = preLayer->getFireCount();
    preFireCount_format = new cuMatrix<int>(preFireCount->rows, preFireCount->cols * preFireCount->channels, 1);
	
    endTime   = Config::instance()->getEndTime(); 
	batch     = Config::instance()->getBatchSize();
	lambda    = Config::instance()->getLambda();
    beta      = Config::instance()->getBeta();
    T_REFRAC  = config->m_t_ref;
    TAU_M     = config->m_tau_m;
    TAU_S     = config->m_tau_s;    

	assert(batch == 1);
	inputSize  = inputs->cols * inputs->channels / endTime;
	outputSize = config->m_numNeurons;

    weightLimit = Config::instance()->getWeightLimit();

    UNDESIRED_LEVEL = config->m_undesired_level;
    DESIRED_LEVEL   = config->m_desired_level;
    MARGIN          = config->m_margin; 

    outputs  = new cuMatrix<bool>(batch, outputSize * endTime, 1);
    outputs_time = new cuMatrix<int>(batch, outputSize * endTime, 1);

    // for fast input response
    inputs_resp_tmp = new cuMatrix<float>(outputSize, endTime * batch, 1);
    inputs_resp = new cuMatrix<float>(batch, outputSize * endTime, 1);
    inputs_float = new cuMatrix<float>(inputSize, endTime * batch, 1);

	curDelta = new cuMatrix<float>(batch, outputSize, 1);
    fireCount= new cuMatrix<int>(batch, outputSize, 1);
    weightSqSum = new cuMatrix<float>(outputSize, 1, 1);
    maxCount    = new cuMatrix<int>(batch, 1, 1);
    accEffect   = new cuMatrix<float>(batch, outputSize * inputSize, 1); 
	reservoirEffect   = new cuMatrix<float>(outputSize, outputSize, 1); 

    predict = NULL;

    // only for the output
    assert(config->m_name != std::string("output"));
    assert(outputSize > 0 && inputSize > 0);

    w        = new cuMatrix<float>(outputSize, inputSize, 1);
    b        = new cuMatrix<float>(outputSize, 1, 1);
    wgrad    = new cuMatrix<float>(outputSize, inputSize, 1);
    bgrad    = new cuMatrix<float>(outputSize, 1, 1);
    wgradTmp = new cuMatrix<float>(outputSize, inputSize, 1);

	if(config->IsReservoirTrain()){
		wgrad_reservoir    = new cuMatrix<float>(outputSize, outputSize, 1);
		wgradTmp_reservoir = new cuMatrix<float>(outputSize, outputSize, 1);
		g1_w_reservoir       = new cuMatrix<float>(outputSize, outputSize, 1); // for adam
		g2_w_reservoir       = new cuMatrix<float>(outputSize, outputSize, 1); // for adam
	}else{
		wgrad_reservoir    = NULL;
		wgradTmp_reservoir = NULL;
		g1_w_reservoir       = NULL; // for adam
		g2_w_reservoir       = NULL; // for adam

	}

	
    assert(config->hasLaterialWeight() == true);
    w_laterial = new cuMatrix<float>(outputSize, outputSize, 1);
    reservoir_connection = new cuMatrix<int>(outputSize, outputSize, 1);
    
    threshold = config->m_vth;
   
    // lateral inihibition factor for the output
    lateralFactor = NULL;
    lateralW = 0.0f;

    assert(Config::instance()->useEffectRatio());
    effectRatio = new cuMatrix<float>(outputSize, inputSize, 1);
	sumEffectRatioInput = new cuMatrix<float>(outputSize, 1, 1);
	sumEffectRatioReservoir = new cuMatrix<float>(outputSize, 1, 1);
    matrixLHS = new cuMatrix<float>(outputSize, outputSize, 1);
    matrixRHS = new cuMatrix<float>(outputSize, inputSize, 1);

    effectPoly = new cuMatrix<float>(50, 5, 1);
	loadPoly("./Effect_Ratio_file/p_Tau_64_600.txt", 50, 5, effectPoly);

    momentum_w = new cuMatrix<float>(outputSize, inputSize, 1);
    momentum_b = new cuMatrix<float>(outputSize, 1, 1);
    g1_w       = new cuMatrix<float>(outputSize, inputSize, 1); // for adam
    g1_b       = new cuMatrix<float>(outputSize, 1, 1);
    g2_w       = new cuMatrix<float>(outputSize, inputSize, 1);
    g2_b       = new cuMatrix<float>(outputSize, 1, 1);
    b1_t = 0.9;
    b2_t = 0.999;
 
	this->initRandom();
    this->initReservoirConnection(config->m_reservoirDim);
    w_ref = NULL;
    w_laterial_ref = NULL;
    b_ref = NULL; 

    if(Config::instance()->getIsGradientChecking())
        this->loadRef(); // for verification purpose

    Layers::instance()->set(m_name, this);
}

void Reservoir::save(FILE* file)
{
    w->toCpu();
    b->toCpu();

    for(int c = 0; c < w->channels; c++){
        for(int i = 0; i < w->rows; i++){
            for(int j = 0; j < w->cols; j++){
                fprintf(file, "%f ", w->get(i, j, c));
            }
        }
    }
    if(w_laterial != NULL){
        for(int c = 0; c < w_laterial->channels; c++){
            for(int i = 0; i < w_laterial->rows; i++){
                for(int j = 0; j < w_laterial->cols; j++){
                    fprintf(file, "%f ", w_laterial->get(i, j, c));
                }
            }
        } 
    }

    for(int c = 0; c < b->channels; c++){
        for(int i = 0; i < b->rows; i++){
            for(int j = 0; j < b->cols; j++){
                fprintf(file, "%f ", b->get(i, j, c));
            }
        }
    }
}

void Reservoir::clearMomentum()
{
    momentum_b->gpuClear();
    momentum_w->gpuClear();
}

//* load the reference weights and output spikes for verification
void Reservoir::loadRef()
{
    if(batch != 1){
        printf("Only do the verification for one batch and one sample!\n");
        exit(0);
    }
    ConfigReservoir * config = (ConfigReservoir*)Config::instance()->getLayerByName(m_name);
    if(config->m_ref_weight_path != std::string("NULL")){
        w_ref = new cuMatrix<float>(outputSize, inputSize, 1);
        initFromDumpfile(config->m_ref_weight_path, w_ref);
        if(config->hasBias()){
            b_ref = new cuMatrix<float>(outputSize, 1, 1);
            initBiasFromDumpfile(config->m_ref_weight_path, b_ref);
        }
    }

    if(config->m_ref_lweight_path != std::string("NULL")){
        w_laterial_ref = new cuMatrix<float>(outputSize, outputSize, 1);
        initFromDumpfile(config->m_ref_lweight_path, w_laterial_ref);
    }

    if(config->m_ref_output_train_path != std::string("NULL")){
        read_each_speech_dump(config->m_ref_output_train_path, output_train_ref, endTime, outputSize);
        assert(output_train_ref.size() == 1 && output_train_ref[0] != NULL);
        output_train_ref[0]->rows = 1;
        output_train_ref[0]->cols = endTime * outputSize;
    }

    if(config->m_ref_output_test_path != std::string("NULL")){
        read_each_speech_dump(config->m_ref_output_test_path, output_test_ref, endTime, outputSize);
        assert(output_test_ref.size() == 1 && output_test_ref[0] != NULL);
        output_test_ref[0]->rows = 1;
        output_test_ref[0]->cols = endTime * outputSize;
   }

}


void Reservoir::initRandom()
{
    ConfigReservoir * config = (ConfigReservoir*)Config::instance()->getLayerByName(m_name);
    float initW = config->m_initW;
 
    if(config->isGaussian()){
        float epsilon = initW;
        for(int c = 0; c < w->channels; c++)
        {
            createGaussian(w->getHost() + c * w->getArea(),
                    outputSize, inputSize, w->channels, epsilon);
        }
        w->toGpu();
    }
    else if(config->isBernoulli()){
        for(int j = 0; j < w->getLen(); j++){
            w->getHost()[j] =  initW * (2.0f * rand() / RAND_MAX - 1.0f);
            //printf("%f ", w->getHost()[j]);
        }//printf("\n");
        w->toGpu();
    }
    else if(config->isFixed()){
        // one input connects to nconnect randomly selected outputs, with initW/-initW
        int nconnect = config->m_weightConnect;
        assert(nconnect > 0);
        for(int c = 0; c < w->channels; ++c){
            for(int i = 0; i < w->rows; ++i){
                for(int t = 0; t < nconnect; ++t){
                    int j = rand() % inputSize;
                    if(rand() % 2 == 0)
                        w->set(i, j, c, initW);
                    else
                        w->set(i, j, c, -1.0*initW);
                    //printf("input_%d to reservoir_%d : %f\n", j, i, w->get(i, j, c));
                }
            }
        }
        w->toGpu();
    }
    else if(config->isExternal()){
        initFromDumpfile(config->m_weightPath, w);
    }
    assert(config->hasLaterialWeight());

}


void Reservoir::initFromCheckpoint(FILE* file)
{
    float val = 0;
    for(int c = 0; c < w->channels; c++){
        for(int i = 0; i < w->rows; i++){
            for(int j = 0; j < w->cols; j++){
                if(fscanf(file, "%f", &val) == EOF)
                {
                    char logStr[256];
                    sprintf(logStr, "scanf fail for layer: %s\n", m_name.c_str());
                    LOG(logStr, "Result/log.txt");
                    assert(0);
                }
                w->set(i, j, c, val);
            }
        }
    }

    if(w_laterial != NULL){
        for(int c = 0; c < w_laterial->channels; c++){
            for(int i = 0; i < w_laterial->rows; i++){
                for(int j = 0; j < w_laterial->cols; j++){
                    if(fscanf(file, "%f", &val) == EOF)
                    {
                        char logStr[256];
                        sprintf(logStr, "scanf fail for layer: %s\n", m_name.c_str());
                        LOG(logStr, "Result/log.txt");
                    }
                    w_laterial->set(i, j, c, val);
                }
            }
        } 
    }

    for(int c = 0; c < b->channels; c++){
        for(int i = 0; i < b->rows; i++){
            for(int j = 0; j < b->cols; j++){
                if(fscanf(file, "%f", &val) == EOF)
                {
                    char logStr[256];
                    sprintf(logStr, "scanf fail for layer: %s\n", m_name.c_str());
                    LOG(logStr, "Result/log.txt");
                    assert(0);
                }
                b->set(i, j, c, val);
            }
        }
    }

    w->toGpu();
    b->toGpu();
}

//* initial the bias weights from the dumped file by the CPU sim
void Reservoir::initBiasFromDumpfile(const std::string& filename, cuMatrix<float>*& cuW)
{
    ifstream f_in(filename.c_str());
    if(!f_in.is_open()){
        printf("Cannot open the file: %s\n", filename.c_str());
        exit(EXIT_FAILURE);
    }
    assert(cuW != NULL);

    int idx; 
    float weight;
    std::string pre_name, post_name;
    while(f_in>>idx>>pre_name>>post_name>>weight){
        int pre = extractNeuronIndex(pre_name);
        int post = extractNeuronIndex(post_name);
        if(pre == inputSize && post < outputSize){ // this is related to bias
            cuW->set(post, 0, 0, weight); 
        }
    }
    cuW->toGpu();
}


//* initial the weights from the dumped file by the CPU sim
void Reservoir::initFromDumpfile(const std::string& filename, cuMatrix<float>*& cuW)
{
    ifstream f_in(filename.c_str());
    if(!f_in.is_open()){
        printf("Cannot open the file: %s\n", filename.c_str());
        exit(EXIT_FAILURE);
    }
 
    assert(cuW != NULL);
    std::vector<std::vector<float> > weights(cuW->rows, std::vector<float>(cuW->cols, 0.0f));
   
    int idx; 
    float weight;
    std::string pre_name, post_name;
    while(f_in>>idx>>pre_name>>post_name>>weight){
        int pre = extractNeuronIndex(pre_name);
        int post = extractNeuronIndex(post_name);
        if(post >= weights.size() || pre >= weights[0].size()){
            if(pre == weights[0].size() && post < weights.size()){ // this is related to bias    
                continue;
            }
            else{
                printf("Read the file: %s, in line: %d\n", filename.c_str(), idx);
                printf("Post: %d, OutputDim: %d\n Pre: %d, InputDim: %d\n", post, (int)weights.size(), pre, (int)weights[0].size());
                assert(post < weights.size() && pre < weights[0].size());
            }
        }
        weights[post][pre] += weight;
    }

    for(int c = 0; c < cuW->channels; c++){
        for(int i = 0; i < cuW->rows; i++){
            for(int j = 0; j < cuW->cols; j++){
                cuW->set(i, j, c, weights[i][j]);
            }
        }
    }
    cuW->toGpu();
    // verify that the weights is correctly copied!
    for(int i = 0; i < weights.size(); ++i){
        for(int j = 0; j < weights[0].size(); ++j){
            assert(fabsf(cuW->get(i, j, 0) - weights[i][j]) < 1e-4);
        }
    }
}


// intialize the reservoir connections
// TODO: improve the randomness of the reservoir (the bad random seed we used now!)
void Reservoir::initReservoirConnection(const std::vector<int>& reservoirDim)
{
    assert(reservoirDim.size() == 3);
    assert(w_laterial != NULL);
    ConfigReservoir * config = (ConfigReservoir*)Config::instance()->getLayerByName(m_name);
    float initW = config->m_initW;
	reservoirSize = 0;
    int d1 = reservoirDim[0], d2 = reservoirDim[1], d3 = reservoirDim[2];
    int num = d1 * d2 * d3;
    if(num != outputSize){
        printf("The reservoir dim: %d x %d x %d = %d does not match the number neuron: %d!\n",d1, d2, d3, num, outputSize);
        exit(EXIT_FAILURE);
    }
    // adopted from the CPU code:
    srand(5);
    std::vector<bool> excitatory(num, false);
    std::vector<dim3> coordinates;
    for(int i = 0; i < excitatory.size(); ++i){
        if(rand() % 100 < 20) excitatory[i] = false;
        else    excitatory[i] = true;
    }
    for(int i = 0; i < d1; ++i){
        for(int j = 0; j < d2; ++j){
            for(int k = 0; k < d3; ++k){
                int index = (i * d2 + j) * d3 + k;
                assert(index < excitatory.size());
                coordinates.push_back(dim3(i, j, k));
            }
        }
    }
    float c;
    float distsq, dist;
    const float factor2 = 1.5;
    for(int j = 0; j < num; ++j){
		int record=0;
		for(int i = 0; i < num; ++i){
            if(excitatory[i]){
                if(excitatory[j]){
                    c = 0.3 * factor2;
                }
                else{
                    c = 0.2 * factor2;
                }
            }
            else{
                if(excitatory[j]){
                    c = 0.4 * factor2;
                }
                else{
                    c = 0.1 * factor2;
                }
            }
            distsq = 0;
            dist = coordinates[i].x -  coordinates[j].x;
            distsq += dist * dist;
            dist = coordinates[i].y -  coordinates[j].y;
            distsq += dist * dist;
            dist = coordinates[i].z -  coordinates[j].z;
            distsq += dist * dist;
            if(rand() % 100000 < 100000 * c * exp(-distsq / 4)){
                w_laterial->getHost()[i + j * outputSize] = initW * (2.0f * rand() / RAND_MAX - 1.0f);
				reservoir_connection->getHost()[record+j*outputSize] = i;
				reservoirSize++;
				record++;
            }
			else{
                w_laterial->getHost()[i + j * outputSize] = 0; // i is input, j is output
			}
        }
		reservoir_connection->getHost()[record+j*outputSize] = -1;
    }
    w_laterial->toGpu();
	reservoir_connection->toGpu();
}

void Reservoir::loadPoly(const std::string& filename, int out_size, int degree, cuMatrix<float>* poly){
    ifstream f_in(filename.c_str());
    if(!f_in.is_open()){
        printf("Cannot open the file: %s\n", filename.c_str());
        exit(EXIT_FAILURE);
    }

	float p;
    std::string data;
	for(int i=0;i<out_size;i++){
		getline(f_in, data);
        std::istringstream iss(data);
		for(int j=0;j<degree;j++){
			iss>>p;
			//std::cout<<ER<<std::endl;
			poly->getHost()[i*degree+j] = p;
		}
	}
    f_in.close();
    poly->toGpu();

}

void Reservoir::verify(const std::string& phrase)
{
    printf("Verify for the layer: %s at %s phrase.\n", m_name.c_str(), phrase.c_str());
    if(phrase == std::string("train"))
    {
        if(!output_train_ref.empty()){
            outputs->toCpu();
            checkMatrixIsSame(output_train_ref[0], outputs, outputSize);
        }
        
    }
    else if(phrase == std::string("test"))
    {
        if(w_ref != NULL){
            w->toCpu();
            checkMatrixIsSame(w_ref, w);
        }
        if(w_laterial_ref != NULL && w_laterial != NULL){
            w_laterial->toCpu();
            checkMatrixIsSame(w_laterial_ref, w_laterial);
        }
 
        if(b_ref != NULL){
            b->toCpu();
            checkMatrixIsSame(b_ref, b);
        }
    
        if(!output_test_ref.empty()){
            outputs->toCpu();
            checkMatrixIsSame(output_test_ref[0], outputs, outputSize);
        }
    }
    printf("Verification for the layer: %s at %s phrase. Pased!!\n", m_name.c_str(), phrase.c_str());
}


/*
 * dim3 block = dim3(batch);
 * dim3 thread= dim3(min(outputSize, 1024));
 */
__global__ void g_Reservoir_feedforward(
    float* inputs_resp,
	float* w,
	float* w_l,
    float* b,
	bool*  outputs,
    int* fireCount,
	int* connection,
	int inputSize,
	int outputSize,
    int endTime,
    float vth,
    int dummyFreq, 
    int T_REFRAC,
    float TAU_M,
    float TAU_S)
{
	int batchId = blockIdx.x;
    int outputSize2 = endTime * outputSize;

	bool* curOutput   = outputs + batchId * outputSize2;
    float* curInput   = inputs_resp + batchId * outputSize2;//inputs_resp:batch * outputSize*endTime 
    int* curFireCount = fireCount + batchId * outputSize; 

    // simulate the spiking train
    for(int tidx = 0; tidx < outputSize; tidx += blockDim.x)
    {
        int o_idx = tidx + threadIdx.x;
        if(o_idx < outputSize)
        {
            float v  = 0.0f;
            float ep = 0.0f;
            float threshold = vth - 1e-6; // migitate the numerical disparity due to fast response
            int t_ref= 0;
            float response = 0.0f;
            int fire_count = 0;
            for(int t = 0; t < endTime; t++){
                // 1. leakage
                v  -= v / TAU_M;
                ep -= ep / TAU_S;
                if(t == 0)
                {
                    curOutput[o_idx + t * outputSize] = false;
                    continue;
                }

                // 2. receive the spike inputs
                __syncthreads(); // make sure all the threads has generated the spikes for the last time step
                response = d_Reservoir_accumulate_spikes(inputSize, outputSize, curInput, curOutput, o_idx, w, w_l, b, t, dummyFreq, endTime, connection);
                
                // 3. Add up the response to ep (state variable)
                ep += response;

                // 4. Update the vmem accordingly
                v += ep/TAU_S;
                if(t_ref > 0){
                    v = 0;
                    t_ref--;
                }
            
                // 5. Fire or not
                curOutput[o_idx + t * outputSize] = v > threshold ?  true : false;
                t_ref = v > threshold ? T_REFRAC : t_ref;
                fire_count += v > threshold ? 1 : 0;
                v = v > threshold ? 0 : v;
            }
            curFireCount[o_idx] = fire_count; 
        }
    }
}



/* the device function to realize: weights * spikes(:, t - 1) + recurrent_weights * o_spikes(t - 1)
 * I only consider the first order dynamics 
 * inputSize  : number of input neurons
 * outputSize : number of output neurons
*/
__device__ float d_Reservoir_accumulate_spikes(
    int inputSize,
    int outputSize,
    float* input_response,
    bool* output,
    int o_idx,
    float* weights,
    float* weights_lat,
    float* biases,
    int t,
    int dummyFreq,
    int endTime,
	int* connection)
{
    int idx = threadIdx.x;
    if(idx >= outputSize * inputSize){
        return 0;
    }  
    float response = 0.0f;
    // effect from the forward-connects
    response = input_response[(t - 1) + o_idx * endTime];

    // effect from the bias
    if(t % dummyFreq == 0){
        response += biases[idx];
    }    

    // effect from the recurrent connections:
    for(int i = 0; i < outputSize; ++i){
		int index = connection[i+o_idx*outputSize];
		if(index < 0)
			break;
		response += output[index + (t - 1) * outputSize] ? weights_lat[index + o_idx * outputSize] : 0;
	}

    return response;
}


/*
 * dim3 block = dim3(batch, inputSize);
 * dim3 thread= min(1024, outputSize);
 */
__global__ void g_Reservoir_synaptic_effect(
        int* inputs_time,
        int* outputs_time,
        int* batchPreFireCount,
        int* batchFireCount,
        float* batchAccEffect,
        int inputSize,
        int outputSize,
        int endTime,
        int T_REFRAC,
        float TAU_M,
        float TAU_S)
{
    int batchId = blockIdx.x;
    int i_idx   = blockIdx.y;

    int wSize        = outputSize * inputSize;
    int inputSize2   = endTime * inputSize;
    int outputSize2  = endTime * outputSize;

    int* input_time       = inputs_time + batchId * inputSize2;
    int* output_time      = outputs_time + batchId * outputSize2;
    int* input_fireCount  = batchPreFireCount + batchId * inputSize;
    int* output_fireCount = batchFireCount + batchId * outputSize;
    float* acc_effect     = batchAccEffect + batchId * wSize;

    for(int i = 0; i < outputSize; i += blockDim.x)
    {
        int o_idx = i + threadIdx.x;
        if(o_idx < outputSize)
        {
            float e = d_Spiking_accumulate_effect_step(output_time, input_time, output_fireCount[o_idx], input_fireCount[i_idx], o_idx, i_idx, outputSize, inputSize, endTime, T_REFRAC, TAU_M, TAU_S);
            acc_effect[i_idx + o_idx * inputSize] = e;
        }
    }
}

/*
 * dim3 block = dim3(batch, outputSize);
 * dim3 thread= min(1024, outputSize);
 */
__global__ void g_Reservoir_synaptic_effect_reservoir(
		float* w_l,
        int* outputs_time,
        int* batchFireCount,
        float* reservoirEffect,
        int outputSize,
        int endTime,
        int T_REFRAC,
        float TAU_M,
        float TAU_S)
{
    int batchId = blockIdx.x;
    int i_idx   = blockIdx.y;

    int outputSize2  = endTime * outputSize;

    int* output_time      = outputs_time + batchId * outputSize2;
    int* output_fireCount = batchFireCount + batchId * outputSize;

    for(int i = 0; i < outputSize; i += blockDim.x)
    {
        int o_idx = i + threadIdx.x;
        if(o_idx < outputSize)
        {
			if(w_l[i_idx + o_idx * outputSize] != 0){
				float e = d_Spiking_accumulate_effect_step(output_time, output_time, output_fireCount[o_idx], output_fireCount[i_idx], o_idx, i_idx, outputSize, outputSize, endTime, T_REFRAC, TAU_M, TAU_S);
				reservoirEffect[i_idx + o_idx * outputSize] = e;
			}else{
				reservoirEffect[i_idx + o_idx * outputSize] = 0;
			}
        }
    }
}

/*
 * dim3 thread = dim3(min(1024, inputSize));
 * dim3 block  = dim3(outputSize);
 */
__global__ void g_Reservoir_sum_effect_ratio_input(
        float* weights,
		float* sumEffectInput,
        int* batchPreFireCount,
        int* batchFireCount,
		float* effectPoly,
		int out_size,
		int degree,
        int inputSize,
		int outputSize,
		float vth)
{
    int batchId = blockIdx.x;
    int o_idx = blockIdx.y;
    int tid     = threadIdx.x;
    extern __shared__ float _sum[];
    _sum[tid] = 0;
    __syncthreads();
    int* input_fireCount  = batchPreFireCount + batchId * inputSize;
    int* output_fireCount   = batchFireCount + batchId * outputSize;

    for(int i = 0; i < inputSize; i += blockDim.x)
    {
        int i_idx = i + tid;
        if(i_idx < inputSize)
        {
			int o_cnt = output_fireCount[o_idx];
            float w = weights[i_idx + o_idx * inputSize];
            //float ratio = i_cnt == 0 || o_cnt == 0 ? 1 : e / float(o_cnt);
            //float ratio = o_cnt == 0 ? 0 : e / float(o_cnt);
            float ratio;
			//if(o_cnt == 0||i_cnt == 0)
			//	ratio=0;
			//else{
				o_cnt = o_cnt > 0? o_cnt : 1;
				o_cnt = o_cnt <= out_size? o_cnt : out_size;
				int i_cnt = input_fireCount[i_idx];
				i_cnt = i_cnt <= out_size? i_cnt : out_size;
				i_cnt = i_cnt > 0? i_cnt : 1;
				ratio=d_Reservoir_Compute_poly(i_cnt, degree, effectPoly, o_cnt);
			//}
            _sum[tid] += w * ratio;
        }
    }
    int len = blockDim.x;
    while(len != 1)
    {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(tid < skip && (tid + skip) < len)
        {
            _sum[tid] += _sum[tid + skip];
        }
        len = skip;
    }
    __syncthreads();
    if(tid == 0){
		sumEffectInput[o_idx] = _sum[0]/vth;
    }
}


/*
 * dim3 thread = dim3(min(1024, outputSize));
 * dim3 block  = dim3(batch,outputSize);
 */
__global__ void g_Reservoir_sum_effect_ratio_reservoir(
        float* w_l,
		float* sumEffectReservoir,
        int* batchFireCount,
		float* effectPoly,
		int out_size,
		int degree,
		int outputSize,
		float vth)
{
    int batchId = blockIdx.x;
    int o_idx = blockIdx.y;
    int tid     = threadIdx.x;
    extern __shared__ float _sum[];
    _sum[tid] = 0;
    __syncthreads();
    int* output_fireCount   = batchFireCount + batchId * outputSize;

    for(int i = 0; i < outputSize; i += blockDim.x)
    {
        int i_idx = i + tid;
        if(i_idx < outputSize)
        {
			int o_cnt = output_fireCount[o_idx];
            float w = w_l[i_idx + o_idx * outputSize];
            //float ratio = i_cnt == 0 || o_cnt == 0 ? 1 : e / float(o_cnt);
            //float ratio = o_cnt == 0 ? 0 : e / float(o_cnt);
			if(w !=0 ){
				float ratio;
				//if(o_cnt == 0)
				//	ratio=0;
				//else{
					o_cnt = o_cnt > 0? o_cnt : 1;
					o_cnt = o_cnt <= out_size? o_cnt : out_size;
					int i_cnt = output_fireCount[i_idx];
					i_cnt = i_cnt <= out_size? i_cnt : out_size;
					i_cnt = i_cnt > 0? i_cnt : 1;
					ratio=d_Reservoir_Compute_poly(i_cnt, degree, effectPoly, o_cnt);
				//}
				_sum[tid] += w * ratio;
			}
        }
    }
    int len = blockDim.x;
    while(len != 1)
    {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(tid < skip && (tid + skip) < len)
        {
            _sum[tid] += _sum[tid + skip];
        }
        len = skip;
    }
    __syncthreads();
    if(tid == 0){
		sumEffectReservoir[o_idx] = _sum[0]/vth;
    }
}



/*
 * dim3 block = dim3(batch, outputSize);
 * dim3 thread= min(1024, outputSize);
 */
__global__ void g_Reservoir_effect_ratio_LHS(
		float* w_l,
        int* batchFireCount,
        float* reservoirEffect,
        float* m_LHS,
		float* sumEffectReservoir,
		float* sumEffectInput,
        int outputSize,
		float vth)
{
    int batchId = blockIdx.x;
    int o_idx   = blockIdx.y;

    int* output_fireCount = batchFireCount + batchId * outputSize;

    for(int i = 0; i < outputSize; i += blockDim.x)
    {
        int i_idx = i + threadIdx.x;
        if(i_idx < outputSize)
        {
			if(i_idx==o_idx){

				float sum = 1 - sumEffectReservoir[o_idx] - sumEffectInput[o_idx];
				if(sum>=0 && sum<1e-4)
					sum=1e-4;
				if(sum<0 && sum>-1e-4)
					sum=-1e-4;
				m_LHS[i_idx + o_idx*outputSize] = sum;
			}else{
				float w = w_l[i_idx + o_idx * outputSize];
				if(w!= 0){
					float e=reservoirEffect[i_idx + o_idx * outputSize];
					int o_cnt = output_fireCount[o_idx];
					int i_cnt = output_fireCount[i_idx];
					float ratio = i_cnt == 0 || o_cnt == 0 ? 0 : e / float(i_cnt);
					m_LHS[i_idx + o_idx*outputSize] = -ratio*w/vth;
				}else{
					m_LHS[i_idx + o_idx*outputSize] = 0;
				}
			}
        }
    }
}

/*
 * dim3 block = dim3(batch, outputSize);
 * dim3 thread= min(1024, inputSize);
 */
__global__ void g_Reservoir_effect_ratio_RHS(
		float* weights,
        int* batchPreFireCount,
        int* batchFireCount,
        float* batchAccEffect,
        float* m_RHS,
		int inputSize,
        int outputSize,
		float vth)
{
    int batchId = blockIdx.x;
    int o_idx   = blockIdx.y;

    int wSize        = outputSize * inputSize;
    int* output_fireCount = batchFireCount + batchId * outputSize;
    int* input_fireCount  = batchPreFireCount + batchId * inputSize;
    float* acc_effect     = batchAccEffect + batchId * wSize;

    for(int i = 0; i < inputSize; i += blockDim.x)
    {
        int i_idx = i + threadIdx.x;
        if(i_idx < inputSize)
        {
            float w = weights[i_idx + o_idx * inputSize];
			float e=acc_effect[i_idx + o_idx * inputSize];
			int o_cnt = output_fireCount[o_idx];
			int i_cnt = input_fireCount[i_idx];
			float ratio = i_cnt == 0 || o_cnt == 0 ? 0 : e / float(i_cnt);
			m_RHS[i_idx + o_idx*inputSize] = w*ratio;
        }
    }
}

void Reservoir::linearSolverQR(cuMatrix<float>* LHS, cuMatrix<float>* RHS, cuMatrix<float>* effectRatio, int lda, int n, int inputSize){
	int bufferSize = 0;
	int *info = NULL;
	float *buffer = NULL;
	float *A = NULL;
	float *tau = NULL;
	int h_info = 0;
	const float one = 1.0;

	cusolverDnHandle_t handle = NULL;
	cublasHandle_t cublasHandle = NULL; // used in residual evaluation
	cudaStream_t stream = NULL;

	//checkCudaErrors(cusolverDnCreate(&handle));
	cusolverDnCreate(&handle);
	checkCudaErrors(cudaStreamCreate(&stream));
	//cudaStreamCreate(&stream);

	//checkCudaErrors(cusolverDnSetStream(handle, stream));
	cusolverDnSetStream(handle, stream);

	checkCudaErrors(cublasCreate(&cublasHandle));
	checkCudaErrors(cublasSetStream(cublasHandle, stream));


	//checkCudaErrors(cusolverDnSgeqrf_bufferSize(
	cusolverDnSgeqrf_bufferSize(
		handle, 
		n, 
		n, 
		LHS->getDev(), 
		lda, 
		&bufferSize);

	checkCudaErrors(cudaMalloc(&info, sizeof(int)));
	checkCudaErrors(cudaMalloc(&buffer, sizeof(float)*bufferSize));
	checkCudaErrors(cudaMalloc(&A, sizeof(float)*lda*n));
	checkCudaErrors(cudaMalloc ((void**)&tau, sizeof(float)*n));

	// prepare a copy of A because potrf will overwrite A with L
	checkCudaErrors(cudaMemcpy(
		A, 
		LHS->getDev(), 
		sizeof(float)*lda*n, 
		cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

	// QR factorization
	//checkCudaErrors(cusolverDnSgeqrf(
	cusolverDnSgeqrf(
		handle, 
		n, 
		n, 
		A, 
		lda,
		tau,
		buffer, 
		bufferSize, 
		info);
	
	checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));
			
	if ( 0 != h_info ){
		printf("%d\n", h_info);
		fprintf(stderr, "Error: LU factorization failed\n");
		assert(0);
	}

	float *x;
	checkCudaErrors(cudaMalloc((void **)&x, sizeof(float)*n*inputSize));

	checkCudaErrors(cudaMemcpy(
		x, 
		RHS->getDev(), 
		sizeof(float)*n*inputSize, 
		cudaMemcpyDeviceToDevice));

		// compute Q^T*b
		//checkCudaErrors(cusolverDnSormqr(
	cusolverDnSormqr(
		handle, 
		CUBLAS_SIDE_LEFT,
		CUBLAS_OP_T,
		n, 
		inputSize, 
		n,
		A,
		lda,
		tau,
		x, 
		n, 
		buffer,
		bufferSize,
		info);

	int info_gpu = 0;

	cudaError_t cudaStat1 = cudaSuccess;
	cudaStat1 = cudaMemcpy(&info_gpu, info, sizeof(int), cudaMemcpyDeviceToHost);
	assert(cudaSuccess == cudaStat1);

	//printf("after ormqr: info_gpu = %d\n", info_gpu);
	assert(0 == info_gpu);

		// x = R \ Q^T*b
	checkCudaErrors(cublasStrsm(
		cublasHandle,
		CUBLAS_SIDE_LEFT,
		CUBLAS_FILL_MODE_UPPER,
		CUBLAS_OP_N,
		CUBLAS_DIAG_NON_UNIT,
		n,
		inputSize,
		&one,
		A,
		lda,
		x,
		n));
		
	checkCudaErrors(cudaDeviceSynchronize());


	checkCudaErrors(cudaMemcpy(
		effectRatio->getDev(), 
		x, 
		sizeof(float)*n*inputSize, 
		cudaMemcpyDeviceToDevice));

		
//test start here, to verify the results
	//float *h_x;
	//h_x = (float*)malloc(sizeof(float)*n*inputSize);
	//checkCudaErrors(cudaMemcpy(h_x, x, sizeof(float)*n*inputSize, cudaMemcpyDeviceToHost));
	//if(checkNan(h_x, n*inputSize)){
	//	float* h_rhs = NULL;
	//	float* h_lhs = NULL;
	//	checkCudaErrors(cudaMemcpy(h_rhs, RHS->getDev(), sizeof(float)*n*inputSize, cudaMemcpyDeviceToHost));
	//	checkCudaErrors(cudaMemcpy(h_lhs, LHS->getDev(), sizeof(float)*n*n, cudaMemcpyDeviceToHost));
	//	printf("\nRHS: \n");
	//	for(int i=0; i< n; i++){
	//		for(int j=0; j<inputSize; j++)
	//			printf("%f\t", h_rhs[j+i*inputSize]);
	//		printf("\n");
	//	}
	//	printf("\n\nLHS: \n");
	//	for(int i=0; i< n; i++){
	//		for(int j=0; j<n; j++)
	//			printf("%f\t", h_lhs[j+i*n]);
	//		printf("\n");
	//	}

	//	if (h_lhs) {
	//		free(h_lhs);
	//	}
	//	if (h_rhs) {
	//		free(h_rhs);
	//	}
	//	
	//	assert(0);
	//}
	//if (h_x) {
	//	 free(h_x);
	//}

	//const float minus_one = -1.0;
	//float *d_A = NULL;
	//float *h_r = NULL; 
	//float *d_r = NULL;
	//float *h_x = NULL; 
	//checkCudaErrors(cudaMalloc((void **)&d_A, sizeof(float)*n*n));

	//checkCudaErrors(cudaMemcpy(
	//	d_A, 
	//	LHS->getDev(), 
	//	sizeof(float)*lda*n, 
	//	cudaMemcpyDeviceToDevice));

	//h_r = (float*)malloc(sizeof(float)*n*inputSize);
	//h_x = (float*)malloc(sizeof(float)*n*inputSize);
	//checkCudaErrors(cudaMalloc((void **)&d_r, sizeof(float)*n*inputSize));
	//checkCudaErrors(cudaMemcpy(d_r, RHS-getDev(), sizeof(float)*n*inputSize, cudaMemcpyDeviceToDevice));


	//// r = b - A*x
	//checkCudaErrors(cublasSgemm_v2(
	//	cublasHandle,
	//	CUBLAS_OP_N,
	//	CUBLAS_OP_N,
	//	n,
	//	inputSize,
	//	n,
	//	&minus_one,
	//	d_A,
	//	lda,
	//	effectRatio->getDev(),
	//	n,
	//	&one,
	//	d_r,
	//	n));

	//checkCudaErrors(cudaMemcpy(h_r, d_r, sizeof(float)*n*inputSize, cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(h_x, x, sizeof(float)*n*inputSize, cudaMemcpyDeviceToHost));

	//printf("\nResidual:\n");
	//vec_norm(n*inputSize, h_r);
	//printf("\nResult:\n");
	//vec_norm(n*inputSize, h_x);


	//if (h_r) {
	//	 free(h_r);
	//}
	//if (d_r) {
	//	 checkCudaErrors(cudaFree(d_r)); 
	//}
	//if (d_A) {
	//	 checkCudaErrors(cudaFree(d_A)); 
	//}
//test end


	if(x){
		checkCudaErrors(cudaFree(x)); 
	}
	if(info){
		checkCudaErrors(cudaFree(info)); 
	}
	if(buffer){
		 checkCudaErrors(cudaFree(buffer)); 
	}
	if(A){
		 checkCudaErrors(cudaFree(A)); 
	}
	if (cublasHandle) {
		 checkCudaErrors(cublasDestroy(cublasHandle));
	}
	if (handle) {
		 cusolverDnDestroy(handle);
	}
	if (tau) {
		 checkCudaErrors(cudaFree(tau));
	}
	if (stream) {
		 checkCudaErrors(cudaStreamDestroy(stream));
	}
}


/*
 * dim3 block = dim3(1);
 * dim3 thread= min(1024, outputSize);
 */
__global__ void g_Reservoir_copy_to_vector(
		float* from,
		float* to,
		int i_idx,
		int outputSize)
{
    for(int i = 0; i < outputSize; i += blockDim.x)
    {
        int o_idx = i + threadIdx.x;
        if(o_idx < outputSize)
        {
			to[o_idx] = from[i_idx + o_idx * outputSize];
        }
	}
}

/*
 * dim3 block = dim3(1);
 * dim3 thread= min(1024, outputSize);
 */
__global__ void g_Reservoir_copy_from_vector(
		float* to,
		float* from,
		int i_idx,
		int outputSize)
{
    for(int i = 0; i < outputSize; i += blockDim.x)
    {
        int o_idx = i + threadIdx.x;
        if(o_idx < outputSize)
        {
			to[i_idx + o_idx * outputSize]= from[o_idx];
        }
	}
}



/*
 * dim3 block = dim3(batch, outputSize);
 * dim3 thread= min(1024, inputSize);
 */
__global__ void g_Reservoir_wgrad_spiketime(
        float* batchAccEffect,
        float* curDelta,
        float* wgradTmp,
		float* sumEffectReservoir,
		float* sumEffectInput,
        int inputSize,
        int outputSize)
{
    int batchId = blockIdx.x;
    int o_idx   = blockIdx.y;
    int tid     = threadIdx.x;
    
    int wSize        = outputSize * inputSize;
    int curDeltaSize = outputSize;

    float* wgrad  = wgradTmp + batchId * wSize;
    float* acc_effect     = batchAccEffect + batchId * wSize;
    float* cDelta = curDelta + batchId * curDeltaSize;

    float delta = cDelta[o_idx];
    for(int i = 0; i < inputSize; i += blockDim.x)
    {
        int i_idx = i + tid;
        if(i_idx < inputSize)
        {
			float sum = 1-sumEffectReservoir[o_idx] - sumEffectInput[o_idx];
			if(sum>=0 && sum<1e-4)
				sum=1e-4;
			if(sum<0 && sum>-1e-4)
				sum=-1e-4;
            float compen_effect = acc_effect[i_idx + o_idx * inputSize]/sum;

            float delta_w = delta * compen_effect;

            wgrad[i_idx + o_idx * inputSize] = delta_w;
        }
    }
}


/*
 * dim3 block = dim3(batch, outputSize);
 * dim3 thread= min(1024, outputSize);
 */
__global__ void g_Reservoir_wgrad_spiketime_reservoir(
		float* w_l,
        float* reservoirEffect,
        float* curDelta,
        float* wgradTmp_reservoir,
		float* sumEffectReservoir,
		float* sumEffectInput,
        int outputSize)
{
    int batchId = blockIdx.x;
    int o_idx   = blockIdx.y;
    int tid     = threadIdx.x;
    
    int curDeltaSize = outputSize;

    float* cDelta = curDelta + batchId * curDeltaSize;

    for(int i = 0; i < outputSize; i += blockDim.x)
    {
        int i_idx = i + tid;
        if(i_idx < outputSize)
        {
			float w = w_l[i_idx + o_idx * outputSize];
			if(w!=0){
				float delta = cDelta[o_idx];
				float sum = 1-sumEffectReservoir[o_idx] - sumEffectInput[o_idx];
				if(sum>=0 && sum<1e-4)
					sum=1e-4;
				if(sum<0 && sum>-1e-4)
					sum=-1e-4;

            	float compen_effect = reservoirEffect[i_idx + o_idx * outputSize]/sum;
            	float delta_w = delta * compen_effect;

            	wgradTmp_reservoir[i_idx + o_idx * outputSize] = delta_w;
			}else{
            	wgradTmp_reservoir[i_idx + o_idx * outputSize] = 0;

			}
        }
    }
}



/*
    block = dim3(1);
    thread = dim3(min(outputSize, 1024));
*/
__global__ void g_Reservoir_setSquareSum(
    float* w_sq_sum,
    int outputSize,
	float v)
{
    int tid = threadIdx.x;

    for(int i = 0; i < outputSize; i += blockDim.x)
    {
        int o_idx = i + tid;
        if(o_idx < outputSize){
			w_sq_sum[o_idx] = v;

		}
	}
}


/*
    block = dim3(1);
    thread = dim3(min(outputSize, 1024));
*/
__global__ void g_Reservoir_divideSquareSum(
    float* w_sq_sum,
    int outputSize,
	int count)
{
    int tid = threadIdx.x;

    for(int i = 0; i < outputSize; i += blockDim.x)
    {
        int o_idx = i + tid;
        if(o_idx < outputSize){
			w_sq_sum[o_idx] /= (float)(count+1);

		}
	}
}

/*
 * block = dim3(outputSize, 1);
 * thread= dim3(min(inputSize, 1024));
*/
__global__ void g_Reservoir_calSquareSum(
    float* w,
    float* w_sq_sum,
    int outputSize,
    int inputSize,
    float weight_limit)
{
    extern __shared__ float _sum[];
    int o_id = blockIdx.x;
    int tid = threadIdx.x;

    _sum[tid] = 0;
    __syncthreads();
    for(int i = 0; i < inputSize; i += blockDim.x)
    {
        int id = i + tid;
        if(id < inputSize)
        { 
            int wid = id + o_id * inputSize;
            float weight = w[wid];
            _sum[tid] += (weight/weight_limit) * (weight/weight_limit);
        }
    }
    __syncthreads();
    int len = blockDim.x;
    while(len != 1)
    {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(tid < skip && (tid + skip) < len)
        {
            _sum[tid] += _sum[tid + skip];
        }
        len = skip;
    }
    if(tid == 0){
        w_sq_sum[o_id] += _sum[0];
	}
}

/*
 * block = dim3(outputSize, 1);
 * thread= dim3(min(outputSize, 1024));
*/
__global__ void g_Reservoir_calSquareSum_reservoir(
    float* w_l,
    float* w_sq_sum,
    int outputSize,
    float weight_limit)
{
    extern __shared__ float _sum[];
    int o_id = blockIdx.x;
    int tid = threadIdx.x;

    _sum[tid] = 0;
    __syncthreads();
    for(int i = 0; i < outputSize; i += blockDim.x)
    {
        int id = i + tid;
        if(id < outputSize)
        { 
            int wid = id + o_id * outputSize;
            float weight = w_l[wid];
			if(weight != 0)
				_sum[tid] += (weight/weight_limit) * (weight/weight_limit);
        }
    }
    __syncthreads();
    int len = blockDim.x;
    while(len != 1)
    {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(tid < skip && (tid + skip) < len)
        {
            _sum[tid] += _sum[tid + skip];
        }
        len = skip;
    }
    if(tid == 0){
        w_sq_sum[o_id] += _sum[0];
	}
}


/*
	block  = dim3(outputSize);
	thread = dim3(inputSize);
*/
__global__ void g_Reservoir_gradAdd(
	float* wgradTmp,
	float* wgrad,
	float* w,
    float* w_sq_sum,
	float lambda,
    float beta,
    float limit,
    int inputSize,
	int outputSize)
{
	int o_idx = blockIdx.x;
	int tid = threadIdx.x;

	for(int i = 0; i < inputSize; i += blockDim.x)
	{
		int i_idx = i + tid;
		if(i_idx < inputSize)
		{
            float wg = wgradTmp[i_idx + o_idx * inputSize];
			float sq_sum = w_sq_sum[o_idx];
			wgrad[i_idx+o_idx*inputSize] = wg + lambda*beta*(w[i_idx+o_idx*inputSize]/limit)*__expf(beta*(sq_sum - 1));
		}
	}
}


/*
	block  = dim3(outputSize);
	thread = dim3(outputSize);
*/
__global__ void g_Reservoir_gradAdd_reservoir(
	float* wgradTmp_reservoir,
	float* wgrad_reservoir,
	float* w_l,
    float* w_sq_sum,
	float lambda,
    float beta,
    float limit,
	int outputSize)
{
	int o_idx = blockIdx.x;
	int tid = threadIdx.x;

	for(int i = 0; i < outputSize; i += blockDim.x)
	{
		int i_idx = i + tid;
		if(i_idx < outputSize)
		{
			float w = w_l[i_idx + o_idx * outputSize];
			if(w!=0){
				float wg = wgradTmp_reservoir[i_idx + o_idx * outputSize];
				float sq_sum = w_sq_sum[o_idx];
				wgrad_reservoir[i_idx+o_idx*outputSize] = wg + lambda*beta*(w/limit)*__expf(beta*(sq_sum - 1));
			}
		}
	}
}


__device__ float d_Reservoir_Compute_poly(
   int i_cnt, 
   int degree, 
   float *effectPoly, 
   int o_cnt)
{
	float result=0;
	for(int i=0; i< degree ; i++){
		float base = (float)o_cnt;
		float exponent = (float)(degree-2-i);
		float coef=(float)(degree-1-i);
		result+=coef*powf(base, exponent)*effectPoly[(i_cnt-1)*degree+i];
	}
	return result;
}

void Reservoir::vec_norm(int n, const float *x)
{
	float norminf = 0;
	float normsum = 0;
	for(int j = 0 ; j < n ; j++){
		float x_abs = fabs(x[j]);
		norminf = (norminf > x_abs)? norminf : x_abs;
		normsum += x_abs;
	}
	printf("inf |b - A*x| = %f \n", norminf);
	printf("sum |b - A*x| = %f \n", normsum);
}

bool Reservoir::checkNan(float* x, int n){
	for(int i=0; i<n; i++){
		if(std::isnan(x[i])){
			return true;
		}
	}
	return false;
}
