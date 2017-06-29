# FaceRecognition
![Alt text](/result/0.jpg)

## Descriptions  
1. Face recognition using __VGG Deep Face Model__. You can find details like performance from the paper, [Deep Face Recognition](https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf).  
  
2. You need to download __model file(.prototxt)__ and __weight file(.caffemodel)__ from [here](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/)  
Go to the link and you can find __Downloads__ section. Download __vgg_face_caffe.tar.gz__ which contains __prototxt file__, __caffemodel file__ and __MATLAB example file__.  

## Requsites  
__1. Caffe (Deep Learning Library)__  
  
__2. openCV (Computer Vision Library)__  

## Directories  
__1. advanced__  
Find the top-5 similar person in the DB with given face image  
  
__2. basic__  
Porting of MATLAB example  

__3. db_example__  
DB image example  
  
__4. img_example__  
Test image example  
  
__5. result__  
Result of __1. advanced__

## How to run  
1. Build and make a exe file. Suppose that __FaceRecognition.exe__.    
2. Open the cmd and put command like  
```FaceRecognition.exe "MODEL_FILE_PATH" "WEIGHT_FILE_PATH" "TEST_IMAGE_FILE_PATH" "DB_DIRECTORY_PATH"```  
