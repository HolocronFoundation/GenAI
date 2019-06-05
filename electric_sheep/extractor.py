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


def __main__():
    video_dir = "/media/troper/Troper_Primary-D/Media/Movies/"
    video_type = ".mp4"
    # TODO: Create a factory method to generate these items
    video_original = {
        "dir": video_dir,
        "name": "Blade Runner (Final Cut)",
        "type": video_type
    }
    video_2049 = {
        "dir": video_dir,
        "name": "Blade Runner 2049",
        "type": video_type
    }
    output_dir = "/media/troper/Troper_Work-DB/electric_sheep/"
    image_type = ".jpg"
    output_original = {
        "dir": output_dir,
        "name": "blade_runner",
        "type": image_type
    }
    output_2049 = {
        "dir": output_dir,
        "name": "blade_runner_2049",
        "type": image_type
    }
    convert_to_image(video_original, output_original)
    convert_to_image(video_2049, output_2049)

if __name__ == '__main__':
    __main__()
