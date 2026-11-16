
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

ORUS_DIR = BASE_DIR / "orus_ocr_v0"
ORUS_CODE = ORUS_DIR / "orus"
ORUS_MODELS = ORUS_DIR / "models"

sys.path.insert(0, str(ORUS_CODE))

from ocr import OCR
from indra_ocr.rec_preproc import crop_line


# --------------------------------------------------
# 2. Streamlit page
# --------------------------------------------------

st.set_page_config(
    page_title="Bangla Handwriting Recognition",
    page_icon="✍️",
    layout="centered"
)

st.title("Bangla Handwriting Recognition")

st.write(
    "Write one Bangla word in the drawing area "
    "and click Recognize Word."
)

st.info(
    "Write one complete word at a time. "
    "Try to keep the letters connected naturally."
)


# --------------------------------------------------
# 3. Load pretrained OCR model
# --------------------------------------------------

@st.cache_resource
def load_ocr():
    if not ORUS_MODELS.exists():
        raise FileNotFoundError(
            f"Orus model directory not found: {ORUS_MODELS}"
        )

    return OCR(
        release=str(ORUS_MODELS),
        hand=False
    )


# --------------------------------------------------
# 4. Drawing canvas
# --------------------------------------------------

st.subheader("Draw your Bangla word")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=8,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=220,
    width=650,
    drawing_mode="freedraw",
    #display_toolbar=True,
    update_streamlit=True,
    return_image_data=True,
    key="bangla_canvas"
)


# --------------------------------------------------
# 5. Recognition
# --------------------------------------------------

if st.button("Recognize Word", type="primary"):

    if canvas_result.image_data is None:

        st.warning("Please write a word first.")

    else:

        try:

            # Convert canvas image to grayscale
            rgba = canvas_result.image_data.astype(np.uint8)

            gray = cv2.cvtColor(
                rgba,
                cv2.COLOR_RGBA2GRAY
            )

            # Find all dark handwritten pixels
            ys, xs = np.where(gray < 245)

            if len(xs) == 0:

                st.warning(
                    "No handwriting detected. Please draw a word."
                )

            else:

                # Find the complete word boundaries
                x0 = int(xs.min())
                x1 = int(xs.max()) + 1
                y0 = int(ys.min())
                y1 = int(ys.max()) + 1

                bbox = [x0, y0, x1, y1]
                st.write("Canvas dimensions:", gray.shape)
                st.write("Detected word bounding box:", bbox)
                st.write("Detected word width:", x1 - x0)
                st.write("Detected word height:", y1 - y0)
                st.write("Detected pixel count:", len(xs))

                with st.spinner(
                    "Recognizing your handwriting..."
                ):

                    # Load OCR model
                    ocr = load_ocr()

                    # Preprocess the entire word
                    # Do not split individual characters
                    line_image = crop_line(
                        gray,
                        bbox,
                        height=ocr.rec.height,
                        window=3,
                        dark=False
                    )
                    st.write("Processed image shape:", line_image.shape)
                    st.write("Minimum pixel value:", float(line_image.min()))
                    st.write("Maximum pixel value:", float(line_image.max()))
                    st.write("Mean pixel value:", float(line_image.mean()))


                    # Display processed images
                    st.image(
                        gray,
                        caption="Original canvas"
                    )

                    st.image(
                        (line_image * 255).astype(np.uint8),
                        caption="Processed image sent to recognizer"
                    )

                    # Predict the complete word
                    predictions = ocr.rec.read_crops(
                        [line_image]
                    )

                    predicted_text, confidence = predictions[0]

                # Display prediction
                if predicted_text.strip():

                    st.success("Recognition completed!")

                    st.subheader("Predicted Word")

                    st.markdown(
                        f"## {predicted_text}"
                    )

                    st.write(
                        f"Recognition confidence: "
                        f"{confidence * 100:.2f}%"
                    )

                else:

                    st.warning(
                        "The model could not recognize this handwriting."
                    )

        except Exception as e:

            st.error(
                f"Recognition error: {str(e)}"
            )

            st.exception(e)


# --------------------------------------------------
# 6. Information
# --------------------------------------------------

with st.expander("About this application"):

    st.write(
        """
        - Task: Bangla handwritten word recognition
        - Recognition: Orus pretrained CTC recognizer
        - Processing: Complete word sequence
        - Character segmentation: Not used
        - Inference: CPU
        """
    )
