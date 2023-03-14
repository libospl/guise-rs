#include "demo/include/blobstore.h"
#include "demo/src/main.rs.h"
#include <algorithm>
#include <functional>
#include <set>
#include <string>
#include <unordered_map>

using namespace guise;

Guise::Guise(std::string resnet, std::string landmk)
{
    detector = dlib::get_frontal_face_detector();
    dlib::deserialize(resnet) >> sp;
    dlib::deserialize(landmk) >> net;
}

// Get the face rectangle and the shape detection.
std::vector<dlib::rectangle> Guise::get_faces(dlib::matrix<dlib::rgb_pixel> img)
{
    return detector(img);
}

// Get retangle coordinates of faces that are the same.
std::map<dlib::rectangle, dlib::rectangle> Guise::compare_images(std::string file_one, std::string file_two)
{
    // The first retangle references file_one, the second references file_two.
    std::map<dlib::rectangle, dlib::rectangle> map;
    // Vector of faces from the first image.
    std::vector<dlib::matrix<dlib::rgb_pixel>> faces;
    // Vector of faces from the second image.
    std::vector<dlib::matrix<dlib::rgb_pixel>> faces2;
    // Image storage.
    dlib::matrix<dlib::rgb_pixel> img;
    dlib::matrix<dlib::rgb_pixel> img2;
    // Temporary retangle data.
    std::vector<dlib::rectangle> tmp;

    // Extract all faces from the first image.
    load_image(img, file_one);
    tmp = get_faces(img);

    // Extract all faces from the second image and compare.
    load_image(img2, file_two);
    for (auto face : get_faces(img2))
    {
        for (long unsigned i = 0; i < tmp.size(); i++)
        {
            std::pair<dlib::rectangle, dlib::rectangle> pair;
            pair.first = tmp[i];
            pair.second = face;
            bool res = compare_face_rectangles(pair, img, img2);

            if (res == true)
            {
                map[pair.first] = pair.second;
                std::cout << "Found pair!" << std::endl;
            }
        }
    }
    return map;
}

// Compare two rectangles.
bool Guise::compare_face_rectangles_files(std::pair<dlib::rectangle, dlib::rectangle> pair, std::string file_one, std::string file_two)
{
    dlib::matrix<dlib::rgb_pixel> img;
    load_image(img, file_one);

    dlib::matrix<dlib::rgb_pixel> img_two;
    load_image(img_two, file_two);

    return compare_face_rectangles(pair, img, img_two);
}

// Compare two rectangles. Please pass the whole image.
bool Guise::compare_face_rectangles(std::pair<dlib::rectangle, dlib::rectangle> pair, dlib::matrix<dlib::rgb_pixel> img_one, dlib::matrix<dlib::rgb_pixel> img_two)
{
    // Extract first face chip using rectangle and image.
    auto val = sp(img_one, pair.first);
    dlib::matrix<dlib::rgb_pixel> face_chip;
    extract_image_chip(img_one, get_face_chip_details(val, 150, 0.25), face_chip);
    face_chip = std::move(face_chip);

    val = sp(img_two, pair.second);
    dlib::matrix<dlib::rgb_pixel> face_chip2;
    extract_image_chip(img_two, get_face_chip_details(val, 150, 0.25), face_chip2);
    face_chip2 = std::move(face_chip2);

    return compare_faces(face_chip, face_chip2);
}

bool Guise::compare_faces(dlib::matrix<dlib::rgb_pixel> &face_one, dlib::matrix<dlib::rgb_pixel> &face_two)
{
    // Convert each face image in faces into a 128D vector.
    std::vector<dlib::matrix<float, 0, 1>> face_descriptors;
    face_descriptors.push_back(net(face_one));
    face_descriptors.push_back(net(face_two));

    // Graph the two faces to see if they are similar enough.
    std::vector<dlib::sample_pair> edges;
    for (size_t i = 0; i < face_descriptors.size(); ++i)
    {
        for (size_t j = i; j < face_descriptors.size(); ++j)
        {
            // If the similarity of the images is close enough, then mark.
            auto len = length(face_descriptors[i] - face_descriptors[j]);
            if (len < .6)
                edges.push_back(dlib::sample_pair(i, j));
        }
    }

    // Number of individials may be less and the number of actual faces in the image.
    std::vector<unsigned long> people; // An array of the two faces identified with an id (a number starting at zero)
    const auto number_of_individuals = chinese_whispers(edges, people);

    if (number_of_individuals == 1)
    {
        return true;
    }
    else
    {
        return false;
    }
}
