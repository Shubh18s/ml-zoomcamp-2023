#!/usr/bin/env python
# coding: utf-8

# In[12]:


get_ipython().run_line_magic('autosave', '0')


# In[23]:


import numpy as np
import tensorflow as tf
from tensorflow import keras


# In[24]:


from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input


# In[25]:


data = "clothing-dataset-small/test"
item = "pants"
url = "c8d21106-bbdb-4e8d-83e4-bf3d14e54c16.jpg"
path = f'{data}/{item}/{url}'

img = load_img(path, target_size=(299,299))


# In[26]:


import numpy as np

x = np.array(img)
X = np.array([x])

X = preprocess_input(X)


# In[27]:


X.shape


# In[28]:


model = keras.models.load_model('xception_v4_41_0.883.h5')


# In[29]:


preds = model.predict(X)


# In[30]:


classes = ['dress',
 'hat',
 'longsleeve',
 'outwear',
 'pants',
 'shirt',
 'shoes',
 'shorts',
 'skirt',
 't-shirt']


# In[31]:


dict(zip(classes, preds[0]))


# ## Convert Keras to TF-Lite

# In[32]:


import tensorflow.lite as tflite

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tf_lite_model = converter.convert()

with open('clothing-model.tflite', 'wb') as f_out:
    f_out.write(tf_lite_model)


# In[33]:


interpreter = tflite.Interpreter(model_path = "clothing-model.tflite")
# loading weights into memory as well for tflite
interpreter.allocate_tensors()


# In[34]:


# finding index for input to model

input_index = interpreter.get_input_details()[0]['index']


# In[35]:


# finding index for output to model

output_index = interpreter.get_output_details()[0]['index']


# In[36]:


interpreter.set_tensor(input_index, X)


# In[37]:


# Now we initialized the input of the interpreter with X 
# Now we need to invoke all the CONVOLUTIONS IN THE NEURAL Network
interpreter.invoke()


# In[38]:


# fetching all results from
interpreter.get_tensor(output_index)


# In[39]:


classes = ['dress',
 'hat',
 'longsleeve',
 'outwear',
 'pants',
 'shirt',
 'shoes',
 'shorts',
 'skirt',
 't-shirt']


# In[1]:


dict(zip(classes, preds[0]))


# ## Removing Tensorflow dependency

# In[41]:


from PIL import Image


# In[42]:


data = "clothing-dataset-small/test"
item = "pants"
url = "c8d21106-bbdb-4e8d-83e4-bf3d14e54c16.jpg"
path = f'{data}/{item}/{url}'


# In[43]:


with Image.open(path) as img:
    img = img.resize((299,299), Image.Resampling.NEAREST)


# In[44]:


img


# In[45]:


# img = load_img(path, target_size=(299,299))


# In[46]:


def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x


# In[47]:


x = np.array(img, dtype='float32')
X = np.array([x])

X = preprocess_input(X)


# In[48]:


interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)


# In[49]:


classes = ['dress',
 'hat',
 'longsleeve',
 'outwear',
 'pants',
 'shirt',
 'shoes',
 'shorts',
 'skirt',
 't-shirt']
dict(zip(classes, preds[0]))


# ## Simpler way of doing it
# ### USe keras-image-helper

# In[3]:


get_ipython().system('pip install keras_image_helper')


# In[23]:


get_ipython().system('pip install tflite-runtime')


# In[6]:


import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor


# In[7]:


data = "clothing-dataset-small/test"
item = "pants"
url = "c8d21106-bbdb-4e8d-83e4-bf3d14e54c16.jpg"
path = f'{data}/{item}/{url}'


# In[8]:


preprocessor = create_preprocessor('xception', target_size=(299, 299))


# In[9]:


# You can also use from_url
X = preprocessor.from_path(path)


# In[10]:


interpreter = tflite.Interpreter(model_path = "clothing-model.tflite")
# loading weights into memory as well for tflite
interpreter.allocate_tensors()

# finding index for input to model

input_index = interpreter.get_input_details()[0]['index']

# finding index for output to model

output_index = interpreter.get_output_details()[0]['index']


# In[11]:


interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)


# In[12]:


classes = ['dress',
 'hat',
 'longsleeve',
 'outwear',
 'pants',
 'shirt',
 'shoes',
 'shorts',
 'skirt',
 't-shirt']
dict(zip(classes, preds[0]))


# In[ ]:





# In[ ]:




