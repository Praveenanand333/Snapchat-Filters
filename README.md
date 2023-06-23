

# Snapchat Filters with Python and OpenCV

This project demonstrates how to apply Snapchat-like filters to images using Python and OpenCV. It allows you to overlay various filters, such as mustaches, hats, and
sunglasses, onto human faces in images and also apply some cool photo effects.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Praveenanand333/Snapchat-Filters.git
```

2. Navigate to the project directory:

```bash
cd Snapchat-Filters
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

Make sure you have Python 3.x and pip installed on your system.

## Usage

1. Place your images for the filters in the `media` directory. Make sure the images have transparent backgrounds or an alpha channel to allow for proper overlaying.

2. Open the `main.py` file and specify the paths to your filter images, such as mustache, hat, or sunglasses.

3. Run the application:

```bash
python main.py
```

4. On the gui that appears select an filter to apply the filters to. The application will detect faces in the image and overlay the chosen filters onto the faces.

5. You can also capture and save images by pressing 'c' and to quit an filter press 'q'

## Customization

You can customize the project according to your needs. Here are some possible modifications:

- Add new filter images: Place new filter images in the `filters` directory and update the `main.py` file with their paths.

- Implement additional filters: Extend the application to include new filter types, such as face masks or funny accessories.

- Adjust filter placement: Modify the code to adjust the position, size, or rotation of the filters based on the detected face landmarks.

## Resources

The project utilizes the following resources:

- OpenCV: https://opencv.org/
- Python: https://www.python.org/
- Sample images: Include a section with credits or references to the sample images used in the project.

## License

[MIT License](LICENSE)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

- Provide acknowledgments or credits to any external libraries, tutorials, or resources that were helpful in the development of the project.

Feel free to modify this README file to suit your specific project requirements and add any additional sections or information as needed.
