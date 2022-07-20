# Masters-Thesis-Implementation

Please run the method "check_constraints" in Project/AutoSimilarityCache/Misc/Misc.py before starting to generate the 
caches in order to detect some errors early.

## Config file Description

### Why you need to read this

This program can run days at a time and can get very memory intensive, especially if many threads are used. Therefore
you need to monitor your memory usage. If you use over 80 % of your memory, page files might be written to
your disk continuously, which can destroy it. As will also be mentioned in more detail further below, the author of this
software is not liable for any damage in connection with the use of this software. Especially if you are on a machine 
with 8GB of memory (such as myself), you might want to leave single-threaded mode set
to true.

Lastly, depending on multiple factors, multithreading does not necessarily bring a performance boost and can even result
in lower performance.

### The parameters
Note: These parameters have to be written in one line only in the config file. (A simple line by line parsing is used in
the first_start() method, if this method is not used then multi-line parameters are not a problem.)

- "first-start": [true, false]
    - Default: true
    - Safety parameter that gets rewritten automatically. Should prevent the user from restarting the dataframe
      generation, which would delete the data.
- "gpu-device": -1 <= gpu-device <= Number of available GPUs -1
    - Default: -1
    - If -1: CPU is used in PyTorch, else the specified GPU is used
    - Mind that the first GPU has the number 0
- "data-generation-mode": ["test", "prod],
    - Default: "prod"
    - If "test" similarities are generated as needed, leading to potentially long delays of requests
    - If "prod" the dataframes have to be already generated
- "single-thread-mode": [true, false],
    - Default: true
    - If true only one thread is used
    - Please read the QA section for further details
- "threads-per-core": 1 <= threads-per-core <= 10,
    - Default: 1
    - Sets the number of threads that can be used per core
- "n-cores": 1 <= n-cores <= physical cores of the CPU
    - Default: 1 
- "overbook-threads": [True, False],
    - Default: False
    - If "True" a thread may register itself as idle and allow another thread to be generated. If the first thread quits
      being idle before the new thread is finished, the maximum thread count is overstepped. Theoretically this option
      makes it possible that the limit is overstepped by 100%.
    - Enabling this option will yield better CPU-Usage, but might be bad if too little memory is available.
- "allow-description-tags": [true, false],
    - Default: false
    - If true, it is allowed to use description tags
    - By default this is set to false since description tags are of very low quality and are scarce, therefore having a
      very significant impact on the performance (and also on the memory requirements)
- "combined-origins-to-consider":
  - Default: ["Title", "Exp", "Title&Exp"]
  - Array that contains combinations of one or more of "Des", "Title", "Exp" connected with "&" in exactly that ordering
    (!) e.g. ["Title&Exp", "Title"] (see section "General Information for Developers") for loading the Dataframes 
    that consider only Title Tags ("Title") and that consider Title Tags and Expert Tags together ("Title&Exp").
  - If a combined origin is loaded in test mode or data generation mode, its subsets must also be loaded.
    This means if 'Des&Title&Exp' is to be generated, the combined origins that are loaded must include:
    ['Exp', 'Title', 'Title&Exp', 'Des', 'Des&Exp', 'Des&Title', 'Des&Title&Exp']
- "average-similarity-weighted-random-sample-size": Dictionary with combined origins as keys and values > 0
  - The average similarity of a tag is calculated using a portion of all tag-origin-tuples, which are shuffled. The 
    probability of a tag-tuple that is randomly drawn to be considered is its weight. (e.g. a tag tuple with weight 0.3 
    has a 70% chance to be discarded.)
- "dynamic-filtering": [true, false]
    - Default: true
    - If true the FilterCache can be used to dynamically filter identifiers for matching
    - This comes at a performance impact, which occurs once everytime the filter is updated.
- "enable-debug-output": [true, false]
  - Default: false
  - Needs to be true that any classes can write debug information into the console
- "enable-process-and-thread-handler-debug-output": [true, false]
  - Default: false
  - If true, the ProcessAndThreadHandler class prints debug information into the console
- "enable-similarity-rank-database-debug-output": [true, false]
  - Default: false
  - If true, the SimilarityRankDatabase class prints debug information into the console
- "enable-matching-debug-output": [true, false]
  - Default: false
  - If true, the matching algorithm prints debug information into the console
- "matching-mode": ["smooth", "experimental"]
  - Default: "smooth"
  - Please refer to the description of the matching algorithms in the document for more details

  
## Questions and Answers:

### How to install the software?

In addition to the Python packages, which where installed using Anaconda for this project, one must install the right
spacy model. After installing the spacy library run the command ```python -m spacy download de_core_news_lg```.

If you want to use an NVIDIA GPU for pytorch first install CUDA (refer to their Website for an installation Guide). Then
refer to the Pytorch Website for the installation of the pytorch library.

The Scipy package may have be installed manually as it is required by numba for linear algebra, but it is nowhere
explicitly imported.

From the reference implementations mentioned on https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
the following files need to be outside the Project folder: coco_eval.py, coco_utils.py, engine.py, transforms.py, 
utils.py

As suggested by https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html the reference implementations
were used for training and validation. Some of the methods in this file contain lines (similar or exact) from or call 
methods of the reference implementations at https://github.com/pytorch/vision/tree/main/references/detection. Thus, as
required by the license, the license text from https://github.com/pytorch/vision/blob/main/LICENSE is included in the
file license_of_reference_implementation.txt.

### First Start

Before running the first_start method in Interface.py, the notebooks "DatasetLoader" and "DatasetCleaning" have to be 
run in that order. Since the program uses Singleton patterns for efficiency, this also means that the Jupyter kernel has
to be manually restarted between running each of these notebooks.
Note that the first_Start method, computes all caches, if this is not what you want you can use the 
programm without executing said method. This means using the software in test mode. In that case however, please read 
the section "General Information for Developers" in this file.

### What options are best in terms of multithreading?

The standard configuration has single threaded mode set to true. The reason for the default
value is that on the one hand more threads will also need more memory and on the other hand, it proved 
to be very difficult to work with dataframes in a threadsafe manner and while a lot of thought went into the locking 
mechanisms, the probability of an error in the results is naturally higher when using multi-threading.

## General Information for Developers

- While the variable name "origin_tuple" refers to tuples containing origins such as ['Exp' , 'Title'], the variable
  name "combined_origin" refers to combinations such as 'Title&Exp'. In the latter case the order is relevant: The
  name 'Exp&Title' is invalid. These are some examples of valid
  elements: ['Exp', 'Title', 'Title&Exp', 'Des', 'Des&Exp', 'Des&Title', 'Des&Title&Exp', 'Icon']
  - 'Icon' may not be used in combination with any other origins. If 'Icon' is loaded, no other combined origins may
     be loaded. The reason behind this is that this origin is only intended to be used for the 
     IconclassComparisonVisualization
- New origins can be added by editing the file ConfigurationMethods.py
- If a new origin is added and caches have already been generated for other origins, run the method new_origin_added. 
  It will adapt some datastructures and delete others, which will be regenerated automatically as needed.
  Do not run code after this method call. When the program is started the next time, initialization processes will 
  recreate removed paths.
  Note: This method is meant to be useful during development. Mind that if the Similarity rank databases for all other 
  origins have been generated before, all entries for the new origin need to be generated as well. This process would
  then have to be done manually. (For a reference for the steps that are involved look at the SimilarityRankDatabase 
  method "__generate_all_similarity_and_rank_databases").
- Some of the cache sizes depend on the "get_max_amount_of_tags_of_loaded_origins" method from Utils.TagProcessing.py.
  Therefore, if the distribution of tags changes in a way that there are outliers with a lot more tags than most other
  identifiers, this method may be changed in order to be more memory efficient.
- If the class "SimilarityAndRankDatabase" is used in test mode (see config-file-parameter "data-generation-mode"), 
  then the database must be saved manually by calling the method "save_generated" of the same class.
- If an origin is added to the get_possible_origins method of ConfigurationMethods.py, then the TagVectorCache and the
  DataAccess Weight Cache must be manually deleted before restarting the program

## Design Decisions
Pandas Dataframes were preferred over traditional databases. Here are the pros and cons:
A plus is that the dataframes exist in memory, therefore the access to it is fast (with very large datasets more memory 
would then be required). A drawback is that they are not threadsafe and therefore more difficult to handle during data-
generation. The locking of entire dataframes for reads and other writes during writes, has probably also a large
performance impact during multi-threaded data generation in the class SimilarityAndRankDatabase. 
The most important thing however was the response time after the dataframes are finished generating, therefore this 
option was chosen.

## Reference Implementaions from PyTorch

As suggested by https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html the reference implementations 
were used for training and validation. Some of these implementations were modified into some methods of the file 
"ObjectDetectionNetworkUtils.py" to better fit the task at hand. Furthermore, the following files of the reference 
implementations are required to run the program: coco_eval.py, coco_utils.py, engine.py, transforms.py, utils.py

## Disclaimer

The author of this thesis and software is in no case liable for any claim, damage or liability that occurs in connection
with dealings of the same. Typos, printing-, software- and typesetting errors in this work reserved. While an effort was
made documenting features, some may remain undocumented. No license can be granted (see document for details). This 
software is hosted on GitHub solely for the purpose of presentation.
