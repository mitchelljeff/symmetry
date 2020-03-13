import tensorflow as tf
import numpy as np
from random import shuffle

batch_size =32
units     = 24
emb_size  = 20
rate      = 0.1
epochs    =1000


vocab={"ga":0, "ti":1, "na":2, "gi":3, "la":4, "li":5, "ni":6, "ta":7, "wo":8, "fe":9, "ko":10, "de":11}

vsize=len(vocab)

seqlen=3

train={"seq":list(), "label":list()}
test ={"seq":list(), "label":list()}

def batcher(data,epochs,batchsize):
    pointers=list(range(len(data["seq"])))
    for e in range(epochs):
        shuffle(pointers)
        for pointer in range(0,len(pointers),batchsize):
            batch={"seq":list(), "label":list()}
            for p in pointers[pointer:pointer+batchsize]:
                batch["seq"].append(data["seq"][p])
                batch["label"].append(data["label"][p])
            yield e,batch,len(batch["seq"])

for split,suffix in [(train,""),(test,"_test")]:
    for form,l in [("abb",1),("aba",0)]:
        with open(form+suffix) as f:
            for line in f:
                seq  =np.zeros((vsize,seqlen), dtype=float)
                #label=np.zeros(0,            dtype=int)
                tokens=line.split()
                for j,token in enumerate(tokens):
                    i=vocab[token]
                    seq[i,j]=1
                    label=l
                split["seq"].append(seq)
                split["label"].append(label)



graph=tf.Graph()

with graph.as_default():
    tfseq  =tf.placeholder(tf.float32,shape=[None,vsize,seqlen])
    tflabel=tf.placeholder(tf.int32,shape=[None])
    tfbs   =tf.placeholder(tf.int32,shape=[])

    w1=tf.get_variable("w1", shape=[vsize+units,units], initializer=tf.contrib.layers.xavier_initializer())
    b1=tf.get_variable("b1", shape=[units], initializer=tf.contrib.layers.xavier_initializer())
    w2=tf.get_variable("w2", shape=[units,2], initializer=tf.contrib.layers.xavier_initializer())
    b2=tf.get_variable("b2", shape=[2], initializer=tf.contrib.layers.xavier_initializer())

    
    state = tf.zeros([tfbs,units],dtype=tf.float32)
    inputs=tf.unstack(tfseq,axis=2)
    outputs=[None]*seqlen

    
    for i in range(seqlen):
        state = tf.concat([inputs[i],state],axis=1)
        state = tf.tanh(tf.matmul(state,w1)+b1)
    logits = tf.matmul(state,w2)+b2

    loss=tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tflabel,logits=logits))
    pred = tf.cast(tf.argmax(logits,axis=1),tf.int32)
    correct=tf.cast(tf.equal(tflabel,pred),tf.float32)
    acc=tf.reduce_sum(correct)

    optimizer = tf.train.AdamOptimizer(rate).minimize(loss)
    
    init = tf.initialize_all_variables()

    saver=tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(init)

    i=0
    n=0
    r=100
    dr=1000
    loss_sum=0
    acc_sum=0
    m_sum=0
    for epoch,batch,bs in batcher(train,epochs,batch_size):
        feed_dict={tfseq:batch["seq"], tflabel:batch["label"], tfbs:bs}
        _, loss_val, acc_val = sess.run([optimizer,loss,acc], feed_dict=feed_dict)

        loss_sum=loss_sum+loss_val
        acc_sum=acc_sum+acc_val
        n=n+bs
        if i%r == 0:
            print("train:",epoch,n,loss_sum/n,acc_sum/n)
            loss_sum=0
            acc_sum=0
            n=0
        if epoch==epochs-1:
            dloss_sum=0
            dacc_sum=0
            dn=0
            for depoch,dbatch,dbs in batcher(test,1,batch_size):
                feed_dict={tfseq:dbatch["seq"], tflabel:dbatch["label"], tfbs:dbs}
                dloss_val, dacc_val = sess.run([loss,acc], feed_dict=feed_dict)
                dloss_sum=dloss_sum+dloss_val
                dacc_sum=dacc_sum+dacc_val
                dn=dn+dbs
            print("***** test:",epoch,dn,dloss_sum/dn,dacc_sum/dn, "*****")
            dloss_sum=0
            dacc_sum=0
            dn=0
        i=i+1



