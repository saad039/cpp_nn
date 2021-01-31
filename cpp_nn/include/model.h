#ifndef MODEL_H
#define MODEL_H

#include "util.h"
#include "tensor.h"

/*

Constants 
====================================================================
num of neurons in all hidden layers                 =  K
number of features per example                      =  F
total examples                                      =  m
classification categories                           =  C
====================================================================

Equations
====================================================================
Z[n] = W[n]A[n-1] + B[n] (Forward Prop) ;         n = layer number
A[n] = g(Z[n])                          ;         g = activation fn  
====================================================================

Matrices Dimensions
====================================================================
Feature Vector(X,A[0])                              = (F,M)
W[1]                                                = (K,F)
W[2]=W[3]=W[4]...W[n-1]                             = (K,K) 
B[1]=B[2]=B[3]...B[n-1]                             = (k,1)
A[1]=A[2]=A[3]...A[n-1]                             = (K,m)
Z[1]=Z[2]=Z[3]...Z[n-1]                             = (K,m)
 
Z[N]                                                = (C,m)
W[n]                                                = (C,K)
B[N]                                                = (C,1)
====================================================================

*/



template<std::size_t hneurons, std::size_t nfeatures, std::size_t nexamples, std::size_t outneurons, std::size_t nlayers>
class model
{

    // typedef K                                           hneurons; //hidden neurons
    // typedef F                                           nfeatures;
    // typedef M                                           nexamples;
    // typedef C                                           outneurons; //output neurons
    // typedef N                                           nlayers;

private:

    tensor_t<nfeatures,nexamples> traindata; //X

    //weights
    tensor_t<hneurons,nfeatures> w1;
    std::array<tensor_t<hneurons,hneurons>,nlayers-2> hweights; //W[2],W[3],W[4],...,[Wnlayers-1]
    tensor_t<outneurons,hneurons> wn;
    
    //biases
    std::array<tensor_t<hneurons,1>,nlayers-1> hbiases; //B[1],B[2],B[3],...,B[nlayers-1]
    tensor_t<outneurons,1> bn;

    //Forward props
    std::array<tensor_t<hneurons,nexamples>,nlayers-1> zn_1s;
    tensor_t<outneurons,nexamples> zn;

    //activations
    std::array<tensor_t<hneurons,nexamples>,nlayers-1> activations;


public:

    model(tensor_t<nfeatures,nexamples>&& data) :traindata(data)
    {}

    constexpr void forward_step() noexcept//performs a single step of forward propagation
    {
        //Z[0] = W1X + B1
        zn_1s[0] = w1.matmul(traindata) + hbiases[0]; //<hneurons,1>

        //A[0] = relu(Z[0])
        activations[0] = zn_1s[0]._transform(util::relu);

//      for(i in range(1,nlayers-2)):
        for(std::size_t i = 0; i < nlayers - 2; i++)
        {
            //  Z[i+1]  = W[i]A[i] + b[i]
            zn_1s[i+1] = hweights[i].matmul(activations[0]) + hbiases[i+1];
            //  A[i+1] = relu(Z[i+1])
            activations[i+1] = zn_1s[i+1]._transform(util::relu);
        }
        //for the final layer
        zn= wn.matmul(activations[activations.size()-1]) + bn;
        
    }

    void compute_loss() //computes loss after a step of forward propagation
    {

    }

    void backprop() //performs a single step of backward propagation
    {

    }

    void summary() const //prints the model's summary to stdout
    {
        using util::println;
        using util::printshape;

        println("total layers: ",nlayers);
        printshape("w1 dims: ",w1.get_shape());
    }

};



#endif


    //Model has layers
    // Each layer has weights
    //Models takes data and performs all the mathematical stuff accross the layers to product a prediction at the final layer
    //It comes up with a prediction, computes the loss.
    //Updates the weights and biases depending upon the loss