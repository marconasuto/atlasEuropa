	?}m4?@?}m4?@!?}m4?@	*??; ??*??; ??!*??; ??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?}m4?@?lu9% ??A???&0?@Y??/?$??rEagerKernelExecute 0*	    ?Q?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator^?I?U@!?V'?X@)^?I?U@1?V'?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?&1???!?q,?_???)?&1???1?q,?_???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismD?l?????!v?4??:??);?O??n??1߹y????:Preprocessing2F
Iterator::Model;?O??n??!߹y????)?~j?t?x?1*M??%|?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap%??C?U@!#Àr?X@)????Mbp?1?_?r?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9)??; ??IG???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?lu9% ???lu9% ??!?lu9% ??      ??!       "      ??!       *      ??!       2	???&0?@???&0?@!???&0?@:      ??!       B      ??!       J	??/?$????/?$??!??/?$??R      ??!       Z	??/?$????/?$??!??/?$??b      ??!       JCPU_ONLYY)??; ??b qG???X@