### 基于rasa的litemind自然语言处理工程

### t1 指代消解 > coref
- 组织方式
    
    
    分词-> 词性标注-> 句法分析 
    实体识别-> 人实体性别判断 -> 指代匹配策略
  
- 目前优化点
    
    
   
    增加人名实体的识别精度, 人名-> 性别

### t2 话单入图 > call2graph
- 组织方式
    
     
     pass 
  
  
- 目前优化点
    
     
     pass

### 问题:

###### 实体识别不够精准

    1) 目前采用ltp提供的单一实体识别;
    2) 缺乏其他广泛实体的识别;
    3) 融合多种实体抽取工具;

###### 姓名属性不够精确
    
    1) 性别判断难度比较大, 模型的准确率有限;
    2) 需要融入规则字典来解决常用名的性别识别;

###### 匹配策略亟待优化  

    1) 采用实体-代词依赖词、上下文相似度;
    2) 需要考虑其他更多可能的情景;


### 安装部署


* 避免多余依赖的引入，最好是用virtualenv创建一个隔离环境

    
    which  python3                              # 查找py3的安装路径
    /usr/local/bin/python3 -m virtualenv .env   # 创建虚拟py3环境


* 获取当前项目的所有依赖，进入隔离环境，将项目的依赖项完整下载：

    
    source .env/bin/activate   # 开启虚拟环境
    pip install pip2pi         # 离线导包工具
    mkdir dependences          # 存放依赖包的地方
    pip2pi ./dependences --no-use-wheel -r requirements.txt  # 根据requirements.txt 导出依赖包

3. 将下载好的依赖项放在服务器，执行命令安装依赖，第一遍安装可能跳过某些依赖项，可以多次执行这个命令：

    
    which  python3                              # 查找py3的安装路径
    /usr/local/bin/python3 -m virtualenv .env   # 创建虚拟py3环境
    pip install --no-index --find-links=./dependences -r requirements.txt  # 新服务器环境下导入依赖包
 
 
### 服务及演示

> python -m server --path models
### 实体抽取
> http://localhost:5000/parse?query=李世民病倒了，小明说他是累病的&project=entity

### 指代消解
> http://localhost:5000/parse?query=李世民病倒了，小明说他是累病的&project=coref

### 属性链接
> http://localhost:5000/parse?query=小明，男，身高180cm，上个月去北京站坐G22到新疆，与他同行的有30岁的小黑，他们开着一辆白色法拉利逃跑&project=link

### 关系抽取
> http://localhost:5000/parse?query=小明生病了，他的阿姨王兰在照顾他&project=relation




### 属性连接
#### 先训练
> python -m train -c sample_configs/config_entity.yml -d data/entity/example -o models/ --project entity
> python -m train -c sample_configs/config_coref.yml -d data/entity/example -o models/ --project coref
> python -m train -c sample_configs/config_link.yml -d data/entity/example -o models/ --project link
> python -m train -c sample_configs/config_relation.yml -d data/entity/example -o models/ --project relation
