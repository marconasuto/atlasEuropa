	7?h????@7?h????@!7?h????@	?P???&???P???&??!?P???&??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:7?h????@? v??y??A'?_[??@Y????ָ?rEagerKernelExecute 0*	gfff."?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??F?6@!???-??X@)??F?6@1???-??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchl#?	???!?zj?)??)l#?	???1?zj?)??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??ݰmQ??!3Æ????)???k???1t?????:Preprocessing2F
Iterator::Model?R\U?]??!?B?k???)N?#Edx?1???????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap='?o|?6@!^3?v?X@)LOX?es?1LU|??d??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?P???&??I?Z?6?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	? v??y??? v??y??!? v??y??      ??!       "      ??!       *      ??!       2	'?_[??@'?_[??@!'?_[??@:      ??!       B      ??!       J	????ָ?????ָ?!????ָ?R      ??!       Z	????ָ?????ָ?!????ָ?b      ??!       JCPU_ONLYY?P???&??b q?Z?6?X@