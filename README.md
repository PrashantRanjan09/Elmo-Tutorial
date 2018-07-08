# Elmo-Tutorial

This is a short tutorial on using Deep contextualized word representations (ELMo) which is discussed in the paper https://arxiv.org/abs/1802.05365.
This tutorial can help in using:

* **Pre Trained Elmo Model** <br>
* **Training an Elmo Model on your new data** <br>
* **Incremental Learning** <br>

While doing Incremental training make sure to put your embedding layer name in line 758:

    exclude = ['the embedding layer name you want to remove']

Visualization of the word vectors using Elmo:

* Tsne
![Optional Text](../master/Tsne_vis.png)

* Tensorboard 
![Optional Text](../master/tensorboard_vis.png)


**Updated changes** :<br>

_train_elmo_updated.py_

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir,restart_ckpt_file)
    
    if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_prefix', help='Prefix for train files')
    parser.add_argument('--restart_ckpt_file', help='latest checkpoint file to start with')
    
This takes an argument (--restart_ckpt_file) to accept the path of the checkpointed file. 


_training_updated.py_


        # load the checkpoint data if needed
        if restart_ckpt_file is not None:
            reader = tf.train.NewCheckpointReader(your_checkpoint_file)
            cur_vars = reader.get_variable_to_shape_map()
            exclude = ['the embedding layer name yo want to remove']
            variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
            loader = tf.train.Saver(variables_to_restore)
            #loader = tf.train.Saver()
            loader.save(sess,'/tmp')
            loader.restore(sess, '/tmp')
            with open(os.path.join(tf_save_dir, 'options.json'), 'w') as fout:
                fout.write(json.dumps(options))

        summary_writer = tf.summary.FileWriter(tf_log_dir, sess.graph)
        
The code reads the checkpointed file and reads all the current variables in the graph and excludes the layers mentioned in the _exclude_ variable, restores rest of the variables along with the associated weights.


### Using Elmo Embedding layer in consequent models
if you want to use Elmo Embedding layer in consequent model build refer : https://github.com/PrashantRanjan09/WordEmbeddings-Elmo-Fasttext-Word2Vec
