	XU/???@XU/???@!XU/???@	d???AR??d???AR??!d???AR??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:XU/???@?]?????A???umߍ@Y??????rEagerKernelExecute 0*	?l?????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?u?;O?Z@!Ѧ??,?X@)?u?;O?Z@1Ѧ??,?X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?HZ????!?????S??)??O??۠?1?d~????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchr?#D??!?vɹO???)r?#D??1?vɹO???:Preprocessing2F
Iterator::Model?L?T???!??E$????)?(??{t?1ň?IDs?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap:ZՒ??Z@!Hw{oh?X@)?l???o?12?;???m?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9d???AR??I??me?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?]??????]?????!?]?????      ??!       "      ??!       *      ??!       2	???umߍ@???umߍ@!???umߍ@:      ??!       B      ??!       J	????????????!??????R      ??!       Z	????????????!??????b      ??!       JCPU_ONLYYd???AR??b q??me?X@