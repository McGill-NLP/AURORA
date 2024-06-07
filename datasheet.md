# Datasheet for dataset "AURORA"

Questions from the [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) paper, v7.

Jump to section:

- [Motivation](#motivation)
- [Composition](#composition)
- [Collection process](#collection-process)
- [Preprocessing/cleaning/labeling](#preprocessingcleaninglabeling)
- [Uses](#uses)
- [Distribution](#distribution)
- [Maintenance](#maintenance)

## Motivation

_The questions in this section are primarily intended to encourage dataset creators
to clearly articulate their reasons for creating the dataset and to promote transparency
about funding interests._

### For what purpose was the dataset created? 

We collected AURORA since there is no current high-quality dataset for instruction-guided image editing where the instruction is an action such as "move carrots into the sink". As a result, the current image editing are quite limited and not as "general" as one would hope. This is an important subtask of image editing and can enable many downstream applications. There have been few training datasets for these sort of edit instructions and the ones that exist have very noisy data, i.e. where the target image shows far more changes than described in the text, or the change described is not even properly shown.

### Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?
It was developed primarily at "Mila - Quebec Artificial Intelligence Institute", specifically in Siva Reddy's lab by his PhD student Benno Krojer. Other collaborators on the paper were involved in the ideation, many of them also at Mila and one of them at Stability AI.

### Who funded the creation of the dataset? 
The dataset was funded by the PI, Siva Reddy.

### Any other comments?
None.

## Composition

_Most of these questions are intended to provide dataset consumers with the
information they need to make informed decisions about using the dataset for
specific tasks. The answers to some of these questions reveal information
about compliance with the EU’s General Data Protection Regulation (GDPR) or
comparable regulations in other jurisdictions._

### What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?

Each datapoint is a triplet of (source image, prompt, target image), i.e. (an image of a dog, "make the dog smile", an image of a dog smiling).
Our data consists of four sub-datasets:
1. MagicBrush: **Source images** are diverse web-scrapped images ([MS-COCO](https://cocodataset.org/#home) which comes from websites like Flickr). **Prompt** and **target images** were previously crowd-sourced with humans using the DALL-E 2 editing interface.
2. Action-Genome-Edit and Something-Something-Edit:** source** and **target images** are video frames depicting activities mostly at home; the **prompt** is human written (crowd-sourced by us or in the case of Something Something at the time of video recording in the original dataset).
3. Kubric-Edit: **Source** and **target images** were generated in a simulation engine (Kubric), and depict non-human objects. The **prompt** is templated.

### How many instances are there in total (of each type, if appropriate)?
9K (MagicBrush) + 11K (Action-Genome-Edit) + 119K (Something-Something-Edit) + 150K (Kubric-Edit) = 149K + 150K = 399K

### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?

It is not really a sample, but we did have to filter out video frames that were too noisy or showed too much change such as camera movement.

### What data does each instance consist of? 

Two raw images (source & target), and a string (the prompt).

### Is there a label or target associated with each instance?

The **target image** is the structure that the model has to predict during training and test time.

### Is any information missing from individual instances?

No.

### Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)?

In MagicBrush there are sequential edits, that are indicated by the json key "img_id".
In Action-Genome edit, some datapoints can come from the same video clip, which can be checked in the filename.

### Are there recommended data splits (e.g., training, development/validation, testing)?

We release training data separately from the AURORA-Bench data. The test data is much smaller and the test split of each of our training sub-datasets contributes to it.

### Are there any errors, sources of noise, or redundancies in the dataset?

The main source of noise comes with the video-frame-based data where sometimes there can be more changes than described in language. Or the change described in the prompt is not shown clearly.

### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?

Self-contained, except that we ask people to download the videos from the original Something Something website, instead of providing the actual image files.

### Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals’ non-public communications)?

No, it was crowd-souced with paid workers who agreed to work on this task.

### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?

No.

### Does the dataset relate to people? 

Especially the Action-Genome-Edit data depicts people in their homes.

### Does the dataset identify any subpopulations (e.g., by age, gender)?

We do not know the exact recruitment for Action-Genome and Something Something videos (we build on top of these), but there are usually requirements such as speaking English.
In our case, we only worked with 7 workers that had shown to produce high-quality data. We do not know their age or other personal details as this information is not direclty shown in Amazon Mechanical Turk.

### Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?

If someone really tried, they might be able to identify some of the people in Action-Genome-Edit since they are shown fully and in their home. This would have to rely on advanced facial recognition and matching with other databases.

### Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?

No.

### Any other comments?
None.

## Collection process

_\[T\]he answers to questions here may provide information that allow others to
reconstruct the dataset without access to it._

### How was the data associated with each instance acquired?

_Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g.,
survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags,
model-based guesses for age or language)? If data was reported by subjects or indirectly
inferred/derived from other data, was the data validated/verified? If so, please describe how._

For the sub-datasets MagicBrush, Action-Genome-Edit and Something-Something-Edit the prompts were written by humans and in the case of MagicBrush, the edited images were also produced in collaboration with an AI editing tool (DALL-E 2).
Only Kubric-Edit is fully synthetic.

The data was verified for "truly minimal" image pairs (as described in the paper), i.e. that all the changes from source to target are also described in the prompt. Only for Something-Something-Edit, this was not done on an instance-level but based on categories/labels and thus there will be some non-minimal pairs.

### What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?

_How were these mechanisms or procedures validated?_

For the data we collected ourselves with humans (Action-Genome-Edit), we used Amazon Mechanical Turk (see Appendix of the paper for screenshots of our interface).
We recruited the best workers from previous test runs and had lengthy e-mail exchanges to verify everything makes sense for them and we get good quality.
For Kubric-Edit we used a simulation engine called Kubric.

### If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?

### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?

We worked with 7 workers from Amazon Mechanical Turk and paid them 0.22$ USD per example, resulting in an estimated 10-20$ per hour.

### Over what timeframe was the data collected?

_Does this timeframe match the creation timeframe of the data associated with the instances (e.g.
recent crawl of old news articles)? If not, please describe the timeframe in which the data
associated with the instances was created._

Around a week at the end of April 2024 for the main collection of Action-Genome. For the others, we constructed it from March to May with refinements.

### Were any ethical review processes conducted (e.g., by an institutional review board)?

No.

### Does the dataset relate to people?

Only in the sense that the prompts were written by people and that 1-2 dataset we build on top of depicts people.

### Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?

We collected it directly from AMT.

### Were the individuals in question notified about the data collection?

_If so, please describe (or show with screenshots or other information) how notice was provided,
and provide a link or other access point to, or otherwise reproduce, the exact language of the
notification itself._

Workers on AMT see the posting with details like price and task description. In our case, we simply emailed workers from previous collections, and also told them it is for a research publication (i.e. linking to similar papers to give them an idea of what they are working on).

### Did the individuals in question consent to the collection and use of their data?

They implicitly agreed to various uses through the terms of service by MTurk: https://www.mturk.com/participation-agreement

### If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?

I am not sure about that part of MTurk's legal agreement but would guess no. I could not find an exact passage describing this, perhaps the the section "Use of Information; Publicity and Confidentiality"

### Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?

No.

### Any other comments?

None.

## Preprocessing/cleaning/labeling

_The questions in this section are intended to provide dataset consumers with the information
they need to determine whether the “raw” data has been processed in ways that are compatible
with their chosen tasks. For example, text that has been converted into a “bag-of-words” is
not suitable for tasks involving word order._

### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?

We mainly filtered out image pairs with too many changes: We told workers to discard images with too many (or in rare cases too few changes). We automatically pre-filtered by "CLIP-score" (the cosine similarity between the visual embeddings of source and target image) for Action-Genome-Edit and Something-Something-Edit.

### Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?

It was not directly saved but can be accessed again by downloading the original sources we build upon such as Action-Genome videos (or frames from [EQBEN](https://github.com/Wangt-CN/EqBen), or the Something Something dataset.

### Is the software used to preprocess/clean/label the instances available?

We provide scripts on how to go from raw videos/frames to the cleaner ones on our repository.

### Any other comments?

None.

## Uses

_These questions are intended to encourage dataset creators to reflect on the tasks
for which the dataset should and should not be used. By explicitly highlighting these tasks,
dataset creators can help dataset consumers to make informed decisions, thereby avoiding
potential risks or harms._

### Has the dataset been used for any tasks already?

We used it to train an image editing model. We expect similar applications, also to video generation models.
MagicBrush has been used by several people.

### Is there a repository that links to any or all papers or systems that use the dataset?

Our code repository, or [MagicBrush](https://github.com/OSU-NLP-Group/MagicBrush).

### What (other) tasks could the dataset be used for?

Training models for video generation, change descriptions (i.e. Vision-and-Language LLMs) or discrimination of two similar images.
A possible negative application further down the road is surveillance systems that need to detect minor changes.

### Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?

Not that I can think of.

### Are there tasks for which the dataset should not be used?

Unsure.

### Any other comments?

None.

## Distribution

### Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? 

We will fully open-source it and provide access via Zenodo/json files as well as Huggingface Datasets.

### How will the dataset will be distributed (e.g., tarball on website, API, GitHub)?

Both Zenodo and Huggingface datasets.

### When will the dataset be distributed?

The weeks after submission to NeurIPS Dataset & Benchmark track, so in June 2024.

### Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?

We stick with the standard open-source license: MIT License

### Have any third parties imposed IP-based or other restrictions on the data associated with the instances?

Something Something is the only dataset with a restricted license (it seems, I don't speak legalese: [License Terms](https://developer.qualcomm.com/software/ai-datasets/something-something)).
So we are planning to link to their website and provide scripts to get to our final data.

### Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?

[Official access and licensing](https://developer.qualcomm.com/software/ai-datasets/something-something) of Something Something dataset.

### Any other comments?

None.

## Maintenance

_These questions are intended to encourage dataset creators to plan for dataset maintenance
and communicate this plan with dataset consumers._

### Who is supporting/hosting/maintaining the dataset?

The main author is responsible for ensuring long-term accessibility, which relies on Zenodo and Huggingface.

### How can the owner/curator/manager of the dataset be contacted (e.g., email address)?

benno.krojer@mila.quebec (or after I finish my PhD benno.krojer@gmail.com)

### Is there an erratum?

Not yet!

### Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?

Not sure yet. If we find that people are interested in the data or trained model, we will continue our efforts.

### If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?

No.

### Will older versions of the dataset continue to be supported/hosted/maintained?

If there ever was an update, yes.

### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?

Since we use a non-restricting license (MIT license), anyone can build on top or include in their training data mixture.

### Any other comments?

No. We hope the data is useful to people!
