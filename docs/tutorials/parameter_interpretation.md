# OpenMIT参数说明


| 参数类型 | 参数名称 | 说明 | 默认值 | 备注 |
| --- | --- | --- | --- | --- |
| 路径信息 | `train_path` | 训练数据路径（目录or文件） | `""` | 训练时必填 |
|  | `valid_path` | 验证数据路径（目录or文件） | `""` | 训练时必填 |
|  | `test_path` | 验证数据路径（目录or文件） | `""` | 预测时必填|
|  | `predict_out` | 预测输出路径 | `""` | 预测时必填| 
|  | `model_dump` | 模型输出路径（明文）| `""` | 训练时必填| 
|  | `model_binary` | 模型输出路径（二进制） | `""` | 训练时必填| 
|  | `model_in` | 模型加载路径 | `""` | 预测时必填| 
| | `data_format` | 数据格式，支持`libsvm/libfm`两种格式 | `libsvm` | |
| | `nbit` | 用于位移操作生成新key（通过feature和feild） | `4` | 当数据格式为libfm时会使用 |
| 任务信息 | `task_type` | 任务类型, train/predict/dump | `"train"` | 
| | `framework` | 分布式计算框架, `mpi/ps` | `ps` | 分布式任务 必填 |
| | `sync_mode` | 参数同步方式, `asp, bsp, ssp` | `asp` | PS框架使用 |
| | `model` | 模型，可选择"lr/fm/ffm/mf" | `"lr"` | 必填 |
| | `optimizer` | 优化器，可选择`gd/adagrad/adadelta/adam/rmsprop`<br>`ftrl/ftml/lbfgs/als`等 | `"ftrl"` | 训练时必填 |
| | `max_epoch` | 最大迭代次数 | `2` | 训练时必须 |
| | `batch_size` | batch大小（样本数） | `100` |  必填 |
| | `max_key` | 最大特征维度id | `<uint64_t>::max()` | optional |
| | `nsample_rate` | 负样本采样率 | `0.0` | 默认不采样 |
| ADMM算法框架 | `rho` | 增广拉格朗日系数（步长）| `1` | MPI框架使用 |
| | `lambda_obj` | 拉格朗日对偶系数 | `0.05` | MPI框架使用 |
| 模型 | `embedding_size` | 隐向量长度 | `4` | FM／FFM模型使用 |
| | `field_combine_set` | 需要做域交叉的field集合，示例`"1,2,3,4"` | `""` | 
| | `field_combine_pair` | 指定feild组合，示例：`1^2,1^3,2^4` | `""` |
| 优化器 | `lr` | 初始学习率 learning rate | `0.1` | 
|  | `l1` | L1正则项系数 | `0.01` | 
|  | `l2` | L2正则项系数 | `0.01` | 
|  | `alpha` | 学习率参数  | `0.01` | 用于FTRL算法 |
|  | `beta` | 学习率参数 | `0.01` | 用于FTRL算法 ｜
|  | `beta1` | 衰减因子 | `0.6` | 用于adam/ftml算法 |
|  | `beta2` | 衰减因子 | `0.99` | 用于adam/ftml算法 |
|  | `gamma` | 衰减因子 | `0.99` | 用于adadelta/rmsprop算法 |
|  | `epsilon` | 常数，防止学习率分母未0 | `1e-8` | 用于防止`动态学习率算法`分母为0 (如`adagrad/adadelta/rmsprop`等) |
| 衡量指标 | `metric` | `auc, logloss, pr` | `auc` | 多个指标选用用逗号分开 | 
| 作业控制 | `trans_level` | 指定事务级别，数值越小说明级别越大 | `1` | 说明leval小于1的事务都会打印 |
| | `is_progress` | 是否打印作业进度信息，`true/false` | `true` | 默认打印 |
| | `job_progress` | 作业进度控制，`batch_size * progress_control`打印进度信息 | `10` | 默认每10个batch打印一条batch信息 |
| Debug | `debug` | 是否debug，会打印出最细粒度信息 | `false` | 日志量会很大，适合小数据集验证 |
| 参数初始化 | `random_name` | 参数随机分布名 | `normal` | 默认正态分布(`normal`)，还支持均匀分布`uniform`等 |
| | `random_mean` | 正态分布均值 | `0.0` | 适用于`random_name=normal` |
| | `random_variance` | 正态分布方差 | `0.01` | 适用于`random_name=normal` | 
| | `random_threshold` | 均匀分布阈值 | `0.01` | 生成`[-threshold, threshold]`之间服从均匀分布的随机值 |
