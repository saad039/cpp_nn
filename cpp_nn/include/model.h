#ifndef MODEL_H
#define MODEL_H

#include<tensor.h>
#include<any>
#include<tuple>
#include<map>

// template<typename precision_type,std::size_t input_dim1, std::size_t input_dim2>
template<typename precision_type, std::size_t input_features,std::size_t train_examples>
class sequential
{
    //Model has layers
    // Each layer has weights
    //Models takes data and performs all the mathematical stuff accross the layers to product a prediction at the final layer
    //It comes up with a prediction, computes the loss.
    //Updates the weights and biases depending upon the loss


};



    // auto model = Model(
    //     input_dims = {},
    //     loss = loss_fn,
    //     train_data =td;
    //     train_labels = tl;
    //     learning_rate = alpha;
    // );

    // auto model = Model(...);
    // model.add_layer(num_neurons,activation=sigmoid) //
    // model.fit(

    // );

#endif