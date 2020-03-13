import tensorflow as tf
import numpy as np
from random import shuffle


longlens=[14,16,18]


def batcher(data,epochs,batchsize):
    pointers=list(range(len(data["seq"])))
    for e in range(epochs):
        shuffle(pointers)
        for pointer in range(0,len(pointers),batchsize):
            batch={"seq":list(), "next":list(), "mask":list(), "slen":list()}
            for p in pointers[pointer:pointer+batchsize]:
                batch["seq"].append(data["seq"][p])
                batch["next"].append(data["next"][p])
                batch["mask"].append(data["mask"][p])
                batch["slen"].append(data["slen"][p])
            yield e,batch,len(batch["seq"])


if __name__ == '__main__':
    
    epochs    =100
    batch_size=128
    units     =100
    stacksize =20
    emb_size  =10
    rate      = 0.001

    max_seq   =40


    vocab={"pad":0, "a":1, "b":2, "c":3, "d":4, "e":5, "o":6}
    v_size=len(vocab)


    train={"seq":list(), "next":list(), "mask":list(), "slen":list()}
    dev={"seq":list(), "next":list(), "mask":list(), "slen":list()}
    test={"seq":list(), "next":list(), "mask":list(), "slen":list()}
    longdev={"seq":list(), "next":list(), "mask":list(), "slen":list()}
    longtest={"seq":list(), "next":list(), "mask":list(), "slen":list()}
    for split,fname in [(train,"abcdtrain"),(dev,"abcddev"),(test,"abcdtest"),(longdev,"longdev"),(longtest,"longtest")]:
        with open(fname) as f:
            for line in f:
                seq=np.zeros((max_seq,), dtype=int)
                next=np.zeros((max_seq,), dtype=int)
                mask=np.zeros((max_seq,), dtype=float)
                slen=np.zeros((1),dtype=int)
                chars=line.split()
                assert len(chars)<max_seq
                slen[0]=(len(chars)-1)/2
                flag=0
                for i,char in enumerate(chars):
                    if char=="o":
                        flag=1
                    seq[i]=vocab[char]
                    mask[i]=flag
                    if i>0:
                        next[i-1]=vocab[char]
                next[i]=vocab["e"]
                split["seq"].append(seq)
                split["next"].append(next)
                split["mask"].append(mask)
                split["slen"].append(slen)

    graph=tf.Graph()

    with graph.as_default():

        tfseq  =tf.placeholder(tf.int32,shape=[None,max_seq])
        tfnext =tf.placeholder(tf.int32,shape=[None,max_seq])
        tfmask =tf.placeholder(tf.float32,shape=[None,max_seq])
        tfslen =tf.placeholder(tf.int32,shape=[None,1])
        tfbs   =tf.placeholder(tf.int32,shape=[])



        e=tf.get_variable("e", shape=[v_size, emb_size], initializer=tf.contrib.layers.xavier_initializer())


        wf=tf.get_variable("wf", shape=[emb_size+units, units], initializer=tf.contrib.layers.xavier_initializer())
        bf=tf.get_variable("bf", shape=[1,units], initializer=tf.contrib.layers.xavier_initializer())

        wi=tf.get_variable("wi", shape=[emb_size+units, units], initializer=tf.contrib.layers.xavier_initializer())
        bi=tf.get_variable("bi", shape=[1,units], initializer=tf.contrib.layers.xavier_initializer())

        wc=tf.get_variable("wc", shape=[emb_size+units, units], initializer=tf.contrib.layers.xavier_initializer())
        bc=tf.get_variable("bc", shape=[1,units], initializer=tf.contrib.layers.xavier_initializer())

        wo=tf.get_variable("wo", shape=[emb_size+units, units], initializer=tf.contrib.layers.xavier_initializer())
        bo=tf.get_variable("bo", shape=[1,units], initializer=tf.contrib.layers.xavier_initializer())

        ws=tf.get_variable("ws", shape=[units, v_size], initializer=tf.contrib.layers.xavier_initializer())
        bs=tf.get_variable("bs", shape=[1,v_size], initializer=tf.contrib.layers.xavier_initializer())


        embedded=tf.gather(e,tfseq)

        h     = tf.zeros([tfbs,units],dtype=tf.float32)
        state = tf.zeros([tfbs,units],dtype=tf.float32)
        inputs=tf.unstack(embedded,axis=1)
        outputs=[None]*max_seq
        for i in range(max_seq):
            hx=tf.concat([h,inputs[i]],axis=1)
            f=tf.sigmoid(tf.matmul(hx,wf)+bf)
            it=tf.sigmoid(tf.matmul(hx,wi)+bi)
            c=tf.tanh(tf.matmul(hx,wc)+bc)
            o=tf.sigmoid(tf.matmul(hx,wo)+bo)
            state = tf.multiply(state,f) + tf.multiply(it,c)
            h = tf.multiply(tf.tanh(state),o)
            outputs[i]=tf.matmul(h,ws)+bs
        logits = tf.stack(outputs,axis=1)

        loss = tf.contrib.seq2seq.sequence_loss(logits,tfnext,tfmask)
        pred = tf.cast(tf.argmax(logits,axis=2),tf.int32)
        correct=tf.multiply(tf.cast(tf.equal(tfnext,pred),tf.float32),tfmask)
        acc=tf.reduce_sum(correct)
        msum=tf.reduce_sum(tfmask)
        longacc=dict()
        longsum=dict()
        for l in longlens:
            lmask=tf.cast(tf.equal(tf.reduce_sum(tfmask,axis=1),tf.constant(float(l+1))),tf.float32)
            longacc[l]=tf.reduce_sum(tf.multiply(tf.reduce_sum(correct,axis=1),lmask))
            longsum[l]=tf.reduce_sum(tf.multiply(lmask,tf.reduce_sum(tfmask,axis=1)))
        accbypos=tf.constant(0)
        optimizer = tf.train.AdamOptimizer(rate).minimize(loss)
        init = tf.initialize_all_variables()

        saver=tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(init)

        i=1
        n=0
        r=100
        dr=1000
        loss_sum=0
        acc_sum=0
        m_sum=0
        for epoch,batch,bs in batcher(train,epochs,batch_size):
            feed_dict={tfseq:batch["seq"], tfnext:batch["next"], tfmask:batch["mask"], tfbs:bs}
            _, loss_val, acc_val, m_val = sess.run([optimizer,loss,acc,msum], feed_dict=feed_dict)

            loss_sum=loss_sum+loss_val
            acc_sum=acc_sum+acc_val
            m_sum=m_sum+m_val
            n=n+bs
            if i%r == 0:
                print("train:",epoch,n,loss_sum/n,acc_sum/m_sum)
                loss_sum=0
                acc_sum=0
                m_sum=0
                n=0
            if (i-r)%dr == 0:
                dloss_sum=0
                dacc_sum=0
                dm_sum=0
                daccbypos_sum=0
                dn=0
                for depoch,dbatch,dbs in batcher(dev,1,batch_size):
                    feed_dict={tfseq:dbatch["seq"], tfnext:dbatch["next"], tfmask:dbatch["mask"], tfbs:dbs}
                    dloss_val, dacc_val, dm_val = sess.run([loss,acc, msum], feed_dict=feed_dict)
                    dloss_sum=dloss_sum+dloss_val
                    dacc_sum=dacc_sum+dacc_val
                    dm_sum=dm_sum+dm_val
                    dn=dn+dbs
                print("***** val:",epoch,dn,dloss_sum/dn,dacc_sum/dm_sum,daccbypos_sum/dn, "*****")
                dloss_sum=0
                dacc_sum=0
                dm_sum=0
                daccbypos_sum=0
                acc14_sum=0
                acc16_sum=0
                acc18_sum=0
                sum14_sum=0
                sum16_sum=0
                sum18_sum=0
                dn=0
                for depoch,dbatch,dbs in batcher(longdev,1,batch_size):
                    feed_dict={tfseq:dbatch["seq"], tfnext:dbatch["next"], tfmask:dbatch["mask"], tfbs:dbs, tfslen:dbatch["slen"]}
                    dloss_val, dacc_val, dm_val, acc14, acc16, acc18, sum14, sum16, sum18 = sess.run([loss,acc, msum, longacc[14], longacc[16], longacc[18], longsum[14], longsum[16], longsum[18]], feed_dict=feed_dict)
                    dloss_sum=dloss_sum+dloss_val
                    dacc_sum=dacc_sum+dacc_val
                    dm_sum=dm_sum+dm_val
                    acc14_sum=acc14_sum+acc14
                    acc16_sum=acc16_sum+acc16
                    acc18_sum=acc18_sum+acc18
                    sum14_sum=sum14_sum+sum14
                    sum16_sum=sum16_sum+sum16
                    sum18_sum=sum18_sum+sum18
                    dn=dn+dbs
                print("***** long:",epoch,dn,dloss_sum/dn,dacc_sum/dm_sum,daccbypos_sum/dn, "*****")
                print("***** 14:",sum14_sum, acc14_sum/sum14_sum, "*****")
                print("***** 16:",sum16_sum, acc16_sum/sum16_sum, "*****")
                print("***** 18:",sum18_sum, acc18_sum/sum18_sum, "*****")
            i=i+1
        dloss_sum=0
        dacc_sum=0
        dm_sum=0
        daccbypos_sum=0
        dn=0
        for depoch,dbatch,dbs in batcher(test,1,batch_size):
            feed_dict={tfseq:dbatch["seq"], tfnext:dbatch["next"], tfmask:dbatch["mask"], tfbs:dbs}
            dloss_val, dacc_val, dm_val = sess.run([loss,acc, msum], feed_dict=feed_dict)
            dloss_sum=dloss_sum+dloss_val
            dacc_sum=dacc_sum+dacc_val
            dm_sum=dm_sum+dm_val
            dn=dn+dbs
        print("***** test:",epoch,dn,dloss_sum/dn,dacc_sum/dm_sum,daccbypos_sum/dn, "*****")
        dloss_sum=0
        dacc_sum=0
        dm_sum=0
        daccbypos_sum=0
        acc14_sum=0
        acc16_sum=0
        acc18_sum=0
        sum14_sum=0
        sum16_sum=0
        sum18_sum=0
        dn=0
        for depoch,dbatch,dbs in batcher(longtest,1,batch_size):
            feed_dict={tfseq:dbatch["seq"], tfnext:dbatch["next"], tfmask:dbatch["mask"], tfbs:dbs, tfslen:dbatch["slen"]}
            dloss_val, dacc_val, dm_val, acc14, acc16, acc18, sum14, sum16, sum18 = sess.run([loss,acc, msum, longacc[14], longacc[16], longacc[18], longsum[14], longsum[16], longsum[18]], feed_dict=feed_dict)
            dloss_sum=dloss_sum+dloss_val
            dacc_sum=dacc_sum+dacc_val
            dm_sum=dm_sum+dm_val
            acc14_sum=acc14_sum+acc14
            acc16_sum=acc16_sum+acc16
            acc18_sum=acc18_sum+acc18
            sum14_sum=sum14_sum+sum14
            sum16_sum=sum16_sum+sum16
            sum18_sum=sum18_sum+sum18
            dn=dn+dbs
        print("***** long:",epoch,dn,dloss_sum/dn,dacc_sum/dm_sum,daccbypos_sum/dn, "*****")
        print("***** 14:",sum14_sum, acc14_sum/sum14_sum, "*****")
        print("***** 16:",sum16_sum, acc16_sum/sum16_sum, "*****")
        print("***** 18:",sum18_sum, acc18_sum/sum18_sum, "*****")
