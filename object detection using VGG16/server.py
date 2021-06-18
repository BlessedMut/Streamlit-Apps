import streamlit as st
import cv2 as cv
import tempfile
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from tensorflow.keras.applications.vgg16 import decode_predictions
# import numpy as np

def upload_video():
    pass

def home():
    pass

# model = VGG16(weights='imagenet', include_top=True)

# def identify(img_path): 

#   img = image.load_img(img_path, color_mode='rgb', target_size=(224,224)) 

#   # Resizing to fit into VGGNET
#   x = image.img_to_array(img)
#   x.shape
#   x = np.expand_dims(x, axis=0)

#   # Using Pre-Trained model for prediction
#   x = preprocess_input(x)
#   features = model.predict(x)
#   p = decode_predictions(features)
#   return p


def play_video(vid_path):
    vf = cv.VideoCapture(vid_path)

    stframe = st.empty()

    font_scale = 3
    font = cv.FONT_HERSHEY_PLAIN

    while vf.isOpened():
        x,y = 0,0
        ret, frame = vf.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv.putText(frame, "[*Object Detected] : Car", (x+10, y+40), font, fontScale=font_scale,color=(0,255,0), thickness=3)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        stframe.image(gray)

def main():
    st.sidebar.title("Dashboard")

    home = st.sidebar.button('Home', key='home')
    upload = st.sidebar.button('Upload video',key='upload')


    if home:

        st.title('Object(s) Detection in video')

        with st.beta_container():

            col1, col2 = st.beta_columns([2,6])

            with col1:
                st.write("Objects in video")
                st.text("1.")

            with col2:
                play_video("cars.mp4")
                # identify(r"C:\Users\Blessed\Downloads\Funny Cat Vines - Short Funny Cats Videos.mp4")
    else:
        f = st.file_uploader("Upload video")

        with st.beta_container():

            col1, col2 = st.beta_columns([2,6])

            with col1:
                st.write("Objects in video")
                st.text("1.")

            with col2:

                if f is not None:
                    tfile = tempfile.NamedTemporaryFile(delete=False) 
                    tfile.write(f.read())


                    vf = cv.VideoCapture(tfile.name)

                    stframe = st.empty()

                    font_scale = 3
                    font = cv.FONT_HERSHEY_PLAIN

                    while vf.isOpened():
                        x,y = 0,0
                        ret, frame = vf.read()
                        # if frame is read correctly ret is True
                        if not ret:
                            print("Can't receive frame (stream end?). Exiting ...")
                            break
                        # p = decode_predictions(identify(frame))
                        # label = identify()[0][0][1]

                        # if (len(p[0][0][0]) != 0):
                        cv.putText(frame, "[*Object Detected] : Car", (x+10, y+40), font, fontScale=font_scale,color=(255,255,0), thickness=3)
                        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                        stframe.image(gray)


    # else:
    #     st.title('Object Detection using VGG16 Model...')

    #     c1, c2, c3 = st.beta_columns([4,2,4])
    #     with c2:
    #         st.write('Done by:')

    #     b1,m1,l1 = st.beta_columns([3,1,3])
    #     with b1:
    #         st.write('Blessed Mutengwa - R182565F')
    #         st.write('blessedmutengwa@gmail.com')
    #     with m1:
    #         st.write("|")
    #     with l1:
    #         st.write('Linval Chisoko - R182565F')
    #         st.write('linvaltchisoko@gmail.com')
    


if __name__ == '__main__':
    main()