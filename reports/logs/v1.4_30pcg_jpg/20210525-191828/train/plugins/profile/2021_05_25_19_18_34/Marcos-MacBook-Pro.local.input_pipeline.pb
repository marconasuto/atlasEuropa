	?:s)??@?:s)??@!?:s)??@	i??%u?i??%u?!i??%u?"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?:s)??@h?o}Xo??As?????@YH?)s????rEagerKernelExecute 0*	U-??P?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??O@_@!?!?a?X@)??O@_@1?!?a?X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?҆?????!'?ܤO???)??RAE՟?1?@?Ǣ??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch4??E`???!Y??2؁??)4??E`???1Y??2؁??:Preprocessing2F
Iterator::ModelM??ӀA??!??Vg??)?fd????1=?#	:?~?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapgI-?_@!??#S?X@)?!S>u?1?O????p?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9i??%u?I\??i??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	h?o}Xo??h?o}Xo??!h?o}Xo??      ??!       "      ??!       *      ??!       2	s?????@s?????@!s?????@:      ??!       B      ??!       J	H?)s????H?)s????!H?)s????R      ??!       Z	H?)s????H?)s????!H?)s????b      ??!       JCPU_ONLYYi??%u?b q\??i??X@