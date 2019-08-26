
import cv2

def convert_to_image(video, output, progress_dot_frequency=240000):
    """Converts a video file to images.

    Parameters
    ----------
    video : dict
        Dictionary of format
            {
                "dir": str,
                    The directory the video is contained in.
                "name": str,
                    The name of the video file.
                "type": str
                    The format of the video file.
            }
    output : type
        Dictionary of format
            {
                "dir": str,
                    The directory to output images to.
                "name": str,
                    The name to use for all files from this run.
                "type": str
                    The type of output.
            }
    progress_dot_frequency : int
        The number of frames at which to print a dot to indicate progress (the default is 240000).

    Returns
    -------
    type
        Description of returned object.

    Raises
    -------
    ExceptionName
        Why the exception is raised.

    """
    source = cv2.VideoCapture(video["dir"] + video["name"] + video["type"])
    success, image = source.read()
    count = 0
    print("Now converting " + video["name"])
    while success:
        cv2.imwrite(output["dir"] + output["name"] + str(count) + output["type"], image)
        success, image = source.read()
        if count % progress_dot_frequency:
            print(".", end="", flush=True)
        count += 1

def process(name, input_dir, input_type, output_dir, output_type):
    """Helper method which formats inputs to convert_to_image.

    Parameters
    ----------
    name : str
        Name of the input file.
    input_dir : str
        Path of the input file directory.
    input_type : str
        Type of input.
    output_dir : str
        Path of the output file directory.
    output_type : type
        Type of output.
    """
    convert_to_image({"dir": input_dir, "name": name, "type": input_type},
                     {"dir": output_dir, "name": name, "type": output_type})

def __main__():
    input_dir = "/media/troper/Troper_Primary-D/Media/Movies/"
    input_type = ".mp4"
    output_dir = "/media/troper/Troper_Work-DB/electric_sheep/"
    output_type = ".jpg"
    to_process = ["Blade Runner (Final Cut)", "Blade Runner 2049"]
    for video in to_process:
        process(video, input_dir, input_type, output_dir, output_type)

if __name__ == '__main__':
    __main__()
