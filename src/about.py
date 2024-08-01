import streamlit as st

from forms.contact import contact_form


@st.experimental_dialog("Contact me")
def show_contact_form():
    contact_form()


# --- HERO SECTION ---
col1, col2 = st.columns(2, gap="small", vertical_alignment="center")
with col1:
    st.image("./assets/san.jpg", width=270)

with col2:
    st.title("Sanyam Sankhala", anchor=False)
    st.write(
        "Fullstack Developer , Robotics and Machine learning Enthusiast!"
    )
    if st.button("✉️ Contact Me"):
        show_contact_form()


# --- EXPERIENCE & QUALIFICATIONS ---
st.write("\n")
st.subheader("Experience & Qualifications", anchor=False)
st.write(
    """
    I am a passionate computer science student at Vellore Institute of Technology, with a keen interest in robotics and mental health technology. My projects range from building self-balancing bots with advanced features like SLAM to creating a comprehensive mental health website named 'Solace Space'. I have experience in various technologies, including OpenCV, Flask, and PaLM, and I actively participate in competitions like Startup-thon. With a background in both hardware and software, I aim to innovate solutions that improve everyday life and well-being. I'm always eager to learn and take on new challenges in the tech world.
    """
)

# --- SKILLS ---
st.write("\n")
st.subheader("Hard Skills", anchor=False)
st.write(
    """
    -Full Stack Development: Proficient in creating dynamic and interactive websites using frontend technologies (HTML, CSS, JavaScript) and backend frameworks like Flask.

    -Robotics and Automation: Designing and building self-balancing bots and home assistance bots with advanced features like SLAM and gesture control.

    -Programming and Software Development: Skilled in Python, C++, and JavaScript for developing applications and implementing complex algorithms.
    -Computer Vision and Object Detection: Utilizing OpenCV and camera modules for real-time object detection and image processing.

    -Embedded Systems: Working with Arduino, ESP32, and Raspberry Pi for various hardware projects involving sensors, motors, and communication modules.

    -Machine Learning and AI: Implementing machine learning models and leveraging tools like PaLM and LangChain for advanced AI applications.
    Project Management and Research: Planning, executing, and documenting complex projects, including writing research papers and presenting innovative solutions in competitions.
    """
)


st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")

# --- HERO SECTION ---
col1, col2 = st.columns(2, gap="small", vertical_alignment="center")
with col1:
    st.image("./assets/aniket12.jpg", width=270)

with col2:
    st.title("Aniket Verma", anchor=False)
    st.write(
        "Fullstack Developer , Robotics and Machine learning Enthusiast!"
    )
    # if st.button("✉️ Contact Me"):
    #     show_contact_form()


# --- EXPERIENCE & QUALIFICATIONS ---
st.write("\n")
st.subheader("Experience & Qualifications", anchor=False)
st.write(
    """
    I'm a passionate student at Vellore Institute of Technology, specializing in Computer Science. With a solid grasp of full-stack development, I thrive on creating seamless user experiences. My enthusiasm for machine learning drives me to explore its applications in innovative ways. I'm dedicated to learning and expanding my skills to contribute meaningfully to the tech industry.
    """
)

# --- SKILLS ---
st.write("\n")
st.subheader("Hard Skills", anchor=False)
st.write(
    """
    -Full Stack Development

    -Programming and Software Development

    -Project Management

    -Embedded Systems

    """
)
