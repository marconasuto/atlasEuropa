	?>V????@?>V????@!?>V????@	9?????9?????!9?????"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?>V????@5??6?N??A??.Q=??@Y?G?z?!@rEagerKernelExecute 0*	    @??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??x?&1Y@!g?^??X@)??x?&1Y@1g?^??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch;?O??n??!?????F??);?O??n??1?????F??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?? ?rh??!??\??B??)????Mb??1?@??>??:Preprocessing2F
Iterator::Model??ʡE???!??0????);?O??n??1?????F??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?????1Y@!?3??X@)y?&1?|?1??/?n|?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no99?????I???	B?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	5??6?N??5??6?N??!5??6?N??      ??!       "      ??!       *      ??!       2	??.Q=??@??.Q=??@!??.Q=??@:      ??!       B      ??!       J	?G?z?!@?G?z?!@!?G?z?!@R      ??!       Z	?G?z?!@?G?z?!@!?G?z?!@b      ??!       JCPU_ONLYY9?????b q???	B?X@