	???uI4?@???uI4?@!???uI4?@	9.?ބv??9.?ބv??!9.?ބv??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:???uI4?@???????A??a??1?@YH??Q???rEagerKernelExecute 0*	??Q?Ծ?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??z<V@!YEQʟ?X@)??z<V@1YEQʟ?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?ξ? =??!?͗V??)?ξ? =??1?͗V??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??#???!x??j??)??À%??1R????:Preprocessing2F
Iterator::Model???`?H??!??7GH??)?mm?y?x?1l?CF?{?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?KK<V@!d????X@){????j?1^?[?En?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no98.?ބv??I?????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??????????????!???????      ??!       "      ??!       *      ??!       2	??a??1?@??a??1?@!??a??1?@:      ??!       B      ??!       J	H??Q???H??Q???!H??Q???R      ??!       Z	H??Q???H??Q???!H??Q???b      ??!       JCPU_ONLYY8.?ބv??b q?????X@