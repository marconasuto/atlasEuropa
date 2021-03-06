?	xz?,?Z?@xz?,?Z?@!xz?,?Z?@		??q{g??	??q{g??!	??q{g??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:xz?,?Z?@????O???A;??l?X?@Yc|??l;??rEagerKernelExecute 0*?x?&???@)      ?=2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?'*?(G@!???@?X@)?'*?(G@1???@?X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???(???!R??TS??)???e????1?H?A????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchD?U??y??!????&??)D?U??y??1????&??:Preprocessing2F
Iterator::Modele?u7??!????	??)e??7it?1
k????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?f??I)G@!?׌???X@)/??$?l?1y??N[3?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??q{g??I?s$?d?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????O???????O???!????O???      ??!       "      ??!       *      ??!       2	;??l?X?@;??l?X?@!;??l?X?@:      ??!       B      ??!       J	c|??l;??c|??l;??!c|??l;??R      ??!       Z	c|??l;??c|??l;??!c|??l;??b      ??!       JCPU_ONLYY??q{g??b q?s$?d?X@Y      Y@q???ơ???"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 