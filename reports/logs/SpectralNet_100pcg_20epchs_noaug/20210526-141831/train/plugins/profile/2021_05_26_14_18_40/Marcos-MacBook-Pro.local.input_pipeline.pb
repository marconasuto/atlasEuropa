	)!XU???@)!XU???@!)!XU???@	??0d????0d??!??0d??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:)!XU???@??hq?0??A??q?E?@Y????????rEagerKernelExecute 0*	    ?MA2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?K7?A?m@!?]??X@)?K7?A?m@1?]??X@:Preprocessing2F
Iterator::ModelˡE?????!????W}??)?z?G???1??у.e??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?Q?????!?W>܍?)?Q?????1?W>܍?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismh??|?5??!Mr?+??)?~j?t???1??? ?y??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap???Mb?m@!W???X@)????Mbp?1Kf??M[?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??0d??I??????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??hq?0????hq?0??!??hq?0??      ??!       "      ??!       *      ??!       2	??q?E?@??q?E?@!??q?E?@:      ??!       B      ??!       J	????????????????!????????R      ??!       Z	????????????????!????????b      ??!       JCPU_ONLYY??0d??b q??????X@