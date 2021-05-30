[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fhate-alert%2FTutorial-ICWSM-2021&count_bg=%2379C83D&title_bg=%23555555&icon=peertube.svg&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

# Hate speech detection, mitigation and beyond (Tutorial ICWSM-2021) 

These are the resources and demos associated with the tutorial *"Hate speech detection, mitigation and beyond"* at [ICWSM 2021](https://www.icwsm.org/2021/index.html). Check our [website](https://hate-alert.github.io/talk/icwsm_tutorial/) here.

**Date and Time** - June 7, 2021, 5:30 PM -7:00 PM IST

# Abstract :bookmark:

Social media sites such as Twitter and Facebook have connected billions of people and given the opportunity to the users to share their ideas and opinions instantly. That being said, there are several ill consequences as well such as online harassment, trolling, cyber-bullying, fake news, and hate speech. Out of these, hate speech presents a unique challenge as it is deep engraved into our society and is often linked with offline violence. Social media platforms rely on local moderators to identify hate speech and take necessary action, but with a prolific increase in such content over the social media many are turning toward automated hate speech detection and mitigation systems. This shift brings several challenges on the plate, and hence, is an important avenue to explore for the computation social science community.

# Contributions and achievements :tada: :tada:

* Our papers are accepted in **top conferences** like AAAI, WWW, CSCW, ICWSM, WebSci. Link to the papers [here](../../tags/our-papers/)
* We have **open sourced** our codes and datasets under a single github organisation - [hate-alert](https://github.com/hate-alert) for the future research in this domain
* We have stored different **transformers models** in [huggingface.co](https://huggingface.co/). Link to [hatealert organisation](https://huggingface.co/Hate-speech-CNERG)
* **Dataset** from our recent accepted paper in AAAI - *"Hatexplain:A Benchmark Dataset for Explainable Hate Speech Detection"* is also stored in the [huggingface datsets forum](https://huggingface.co/datasets/hatexplain)
* We also participate in several hate speech shared tasks, winning many of them - [hatealert@DLTEACL](https://www.aclweb.org/anthology/2021.dravidianlangtech-1.17.pdf), [hateminers@AMI](http://personales.upv.es/prosso/resources/FersiniEtAl_Evalita18.pdf), [hatemonitors@HASOC](https://dl.acm.org/doi/10.1145/3368567.3368584) and coming under 1% in [hatealert@Hatememe detection](https://www.drivendata.org/competitions/70/hateful-memes-phase-2/leaderboard/) by Facebook AI.   
* [Notion page](https://www.notion.so/punyajoy/Hate-speech-papers-resource-7fc20fa1bea64cbdb30862092ae197b3) containing hate speech papers. 

# Few demos :abacus:

We also provide some demos for the social scientists so that our opensource models can be used. Please provide feedback in the [issues](https://github.com/hate-alert/Tutorial-ICWSM-2021/issues).

* **Multlingual abuse predictor** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hate-alert/Tutorial-ICWSM-2021/blob/main/Demos/Multilingual_abuse_predictor.ipynb) - This presents a suite of models which try to predict abuse in different languages. Different models are built upon the dataset found from that language. You can upload a file in the specified format and get back the predicitions of these models. 
* **Rationale predictor demo** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hate-alert/Tutorial-ICWSM-2021/blob/main/Demos/Rationale_predictor_demo.ipynb) - This is a model trained using rationale and classifier head. Along with predicting the abusive or non-abusive label, it can also predict the rationales i.e. parts of text which are abusive according to the model.


> :rotating_light: **Check the individual colab demos to learn more about the how to use these tools. These models might carry potential biases, hence should be used with appropriate caution.** :rotating_light:

###  :thumbsup: The repo is still in active developements. Feel free to create an [issue](https://github.com/hate-alert/Tutorial-ICWSM-2021/issues) for the demos as well as the notion page that we shared!!  :thumbsup:

