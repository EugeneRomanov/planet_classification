## Introduction
This project was created to organize a pipeline model for a multilabel classification dataset “Planet: Understanding the Amazon from Space”.

Every minute, the world loses an area of forest the size of 48 football fields. And deforestation in the Amazon Basin accounts for the largest share, contributing to reduced biodiversity, habitat loss, climate change, and other devastating effects. But better data about the location of deforestation and human encroachment on forests can help governments and local stakeholders respond more quickly and effectively.

Planet, designer and builder of the world’s largest constellation of Earth-imaging satellites, will soon be collecting daily imagery of the entire land surface of the earth at 3-5 meter resolution. 
While considerable research has been devoted to tracking changes in forests, it typically depends on course-resolution imagery from Landsat (30 meter pixels) or MODIS (250 meter pixels). This limits its effectiveness in areas where small-scale deforestation or forest degradation dominate.

Furthermore, these existing methods generally cannot differentiate between human causes of forest loss and natural causes. Higher resolution imagery has already been shown to be exceptionally good at this, but robust methods have not yet been developed for Planet imagery.

Our task is to create a multi-class classification model that can label satellite images with different classes of land cover and land use. The model will help to understand where, how and why deforestation occurs in the world and how to respond to it.

The project consists of two parts: modeling and service.
