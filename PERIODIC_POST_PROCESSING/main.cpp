/**
 * Post mission processing
 * @author Diego Encabo del Castillo
 */

#include <iostream>
#include <unistd.h>
#include <stdint.h>
#include <err.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cmath>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"

constexpr int8_t ARGC = 2;
constexpr int8_t ANY_KEY = 0;
constexpr int8_t NUM_COLUMNS = 10;
constexpr int8_t RGB_NAME_INDEX = 0;
constexpr int8_t IR_NAME_INDEX = 1;
constexpr int8_t X_INDEX = 2;
constexpr int8_t Y_INDEX = 3;
constexpr int8_t Z_INDEX = 4;
constexpr int8_t YAW_INDEX = 9;
const std::string HELP_ARG = "--help";
const std::string HELP_FLAG = "-h";
const std::string SLASH_STR = "/";
const std::string CSV_FILENAME = "metadata.csv";
constexpr char COMMA_DELIM = ',';
constexpr int32_t MAX_FEATURES = 7000;
constexpr float GOOD_MATCH_PERCENT = 0.30f;
constexpr int32_t RED_CHANNEL_INDEX = 2;
constexpr int32_t GREEN_CHANNEL_INDEX = 1;
constexpr int32_t BLUE_CHANNEL_INDEX = 0;
constexpr int32_t OTSU_MIN_THRESHOLD = 0;
constexpr int32_t OTSU_MAX_THRESHOLD = 255;
constexpr int32_t MIN_PIXEL_VALUE = 0;
constexpr int32_t MAX_PIXEL_VALUE = 255;
constexpr int32_t MIN_INDEX_THRESHOLD = 0;
constexpr int32_t MAX_INDEX_THRESHOLD = 1;
constexpr int32_t SQUARE_SIZE = 40;


cv::Mat registerIrImage(const cv::Mat &rgbImg, const cv::Mat &irImg)
{
  cv::Mat rgbImgGray;

  cv::cvtColor(rgbImg, rgbImgGray, cv::COLOR_BGR2GRAY);

  // Variables to store keypoints and descriptors
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  cv::Mat descriptors1, descriptors2;

  // Detect ORB features and compute descriptors.
  cv::Ptr<cv::Feature2D> orb = cv::ORB::create(MAX_FEATURES);
  orb->detectAndCompute(irImg, cv::Mat(), keypoints1, descriptors1);
  orb->detectAndCompute(rgbImgGray, cv::Mat(), keypoints2, descriptors2);


  // Match features.
  std::vector<cv::DMatch> matches;
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
  matcher->match(descriptors1, descriptors2, matches, cv::Mat());

  // Sort matches by score
  std::sort(matches.begin(), matches.end());

  // Remove not so good matches
  const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
  matches.erase(matches.begin()+numGoodMatches, matches.end());

// Draw top matches
  cv::Mat imMatches;
  cv::drawMatches(irImg, keypoints1, rgbImg, keypoints2, matches, imMatches);

  // Extract location of good matches
  std::vector<cv::Point2f> points1, points2;

  for( size_t i = 0; i < matches.size(); ++i)
  {
    points1.push_back( keypoints1[ matches[i].queryIdx ].pt );
    points2.push_back( keypoints2[ matches[i].trainIdx ].pt );
  }

  // Find homography
  cv::Mat h = cv::findHomography(points1, points2, cv::RANSAC);

  cv::Mat registeredImg;

  // Use homography to warp image
  cv::warpPerspective(irImg, registeredImg, h, rgbImg.size());

  return registeredImg;
}

struct WAYPOINT_DATA
{
  cv::Mat rgbImg;
  std::string rgbImgName;
  cv::Mat irImg;
  std::string irImgName;
  double x;
  double y;
  double z;
  double yaw;

  WAYPOINT_DATA(const cv::Mat &rgbImg_,
                const std::string &rgbImgName_,
                const cv::Mat &irImg_,
                const std::string &irImgName_,
                const double x_,
                const double y_,
                const double z_,
                const double yaw_)
  {
    rgbImg = rgbImg_;
    rgbImgName = rgbImgName_;
    irImg = irImg_;
    irImgName = irImgName_;
    x = x_;
    y = y_;
    z = z_;
    yaw = yaw_;


    printf("ALIGNING IMAGES... %s AND %s\n", rgbImgName.c_str(), irImgName.c_str()); 
    irImg = registerIrImage(rgbImg, irImg);

    cropImages();
  }

  void cropImages()
  {
    rgbImg = rgbImg(cv::Range(0, rgbImg.rows), cv::Range(200, rgbImg.cols));
    irImg = irImg(cv::Range(0, irImg.rows), cv::Range(200, irImg.cols));
  }


};



void printUsage(const bool help=false)
{
  if(true == help)
  {
    std::cout << "usage: post_mission_processing [mission_dir]" << std::endl;
    std::cout << "Note: mission_dir is the directory auto-generated during the mission." 
    "It must contain a CSV file named 'metadata.csv' with columns: "
          "num_waypoint,rgb_image_name,ir_image_name,x,y,z,lat,lon,yaw." << std::endl;
  }
  else
  {
    std::cout << "usage: post_mission_processing [-h/--help] for more detailed usage." << std::endl;
  }
}

void checkDirectory(const char *path)
{
  const std::filesystem::path dirPath = path;
  const std::string csvFilePathStr = path + SLASH_STR + CSV_FILENAME;

  if(false == std::filesystem::is_directory(dirPath))
  {
    errx(EXIT_FAILURE, "'%s' is not a directory!", path);
  }
  else
  {
    const std::filesystem::path csvFilePath(csvFilePathStr);

    if(false == std::filesystem::exists(csvFilePath))
    {
      errx(EXIT_FAILURE, "'%s' does not exist inside mission_dir!", CSV_FILENAME.c_str());
    }
  }

}

void checkArgs(int32_t argc, char * argv[])
{
  if(argc != ARGC)
  {
    printUsage();
    std::exit(EXIT_FAILURE);
  }
  else
  {
    if((HELP_ARG == argv[1]) || (HELP_FLAG == argv[1]))
    {
      printUsage(true);
      std::exit(EXIT_SUCCESS);
    }
    else
    {
      checkDirectory(argv[1]);
    }
  }
}

std::vector<WAYPOINT_DATA> initWaypointsData(const char *dirPath)
{
  std::vector<WAYPOINT_DATA> retList;
  const std::string csvFilePath = dirPath + SLASH_STR + CSV_FILENAME;

  std::ifstream csvFile(csvFilePath);

  if(true == csvFile.is_open())
  {
    std::string csvLine;
    
    // We discard first row since it is the column names
    std::getline(csvFile, csvLine);

    // Iterate each line of CSV file
    while(std::getline(csvFile, csvLine))
    {
      std::stringstream ss(csvLine);
      std::array<std::string,NUM_COLUMNS> rowValues;
      std::string token;
      int32_t i = 0;

      //  Iterate each token of CSV file's line delimited by comma
      while(std::getline(ss, token, COMMA_DELIM))
      {
        rowValues[i++] = token;
      }

      const std::string rgbImagePath = dirPath + SLASH_STR + rowValues[RGB_NAME_INDEX];
      const std::string irImagePath = dirPath + SLASH_STR + rowValues[IR_NAME_INDEX];

      const cv::Mat rgbImage = cv::imread(rgbImagePath);
      const cv::Mat irImage = cv::imread(irImagePath);

      if((false == rgbImage.empty()) && (false == irImage.empty()))
      {
        const double x = std::atof(rowValues[X_INDEX].c_str());
        const double y = std::atof(rowValues[Y_INDEX].c_str());
        const double z = std::atof(rowValues[Z_INDEX].c_str());
        const double yaw = std::atof(rowValues[YAW_INDEX].c_str());

        const WAYPOINT_DATA auxStruct(
          rgbImage, rowValues[RGB_NAME_INDEX], 
          irImage, rowValues[IR_NAME_INDEX], 
          x, y, z, yaw);
        
        retList.push_back(auxStruct);
      }
      else
      {
        if(true == rgbImage.empty())
        {
          errx(EXIT_FAILURE, "Could not open '%s'", rgbImagePath.c_str());
        }

        if(true == irImage.empty())
        {
          errx(EXIT_FAILURE, "Could not open '%s'", irImagePath.c_str());
        }
      }
    }
  }
  else
  {
    errx(EXIT_FAILURE, "Could not open CSV file '%s'!", csvFilePath.c_str());
  }

  return retList;
}


void applyARVI(cv::Mat rgb_im, const cv::Mat &ir_im, cv::Mat &arvi_im)
{
  std::vector<cv::Mat> rgb_channels;
  std::vector<cv::Mat> arvi_channels;

  cv::split(rgb_im, rgb_channels);
  cv::split(arvi_im, arvi_channels);

  float NIR, RED, BLUE, ARVI;

  for(int i = 0; i < rgb_im.rows; ++i){
    for(int j = 0; j < rgb_im.cols; ++j){
      NIR = ir_im.at<uchar>(i, j);
      RED = rgb_channels[RED_CHANNEL_INDEX].at<uchar>(i, j);
      BLUE = rgb_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j);

      ARVI = (NIR - (2 * RED) + BLUE) / (NIR + (2 * RED) + BLUE);


      if(ARVI <= 0)
      {
        arvi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        arvi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 0;
        arvi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0 < ARVI <= 0.25)
      {
        arvi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        arvi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 128;
        arvi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0.25 < ARVI <= 0.5)
      {
        arvi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        arvi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 255;
        arvi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0.5 < ARVI <= 0.75)
      {
        arvi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        arvi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 255;
        arvi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 0;
      }
      else
      {
        arvi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        arvi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 128;
        arvi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 0;
      }
    }
  }

  cv::merge(arvi_channels, arvi_im);
}


void applyEVI(cv::Mat rgb_im, const cv::Mat &ir_im, cv::Mat &evi_im)
{
  std::vector<cv::Mat> rgb_channels;
  std::vector<cv::Mat> evi_channels;

  cv::split(rgb_im, rgb_channels);
  cv::split(evi_im, evi_channels);

  float NIR, RED, BLUE, EVI;
  float G = 2.5f;
  float C1 = 6.0f;
  float C2 = 4.5f;
  float L = 1.0f;

  for(int i = 0; i < rgb_im.rows; ++i){
    for(int j = 0; j < rgb_im.cols; ++j){
      NIR = ir_im.at<uchar>(i, j);
      RED = rgb_channels[RED_CHANNEL_INDEX].at<uchar>(i, j);
      BLUE = rgb_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j);

      EVI =  G * (NIR -RED) / (NIR + C1 * RED - C2 * BLUE + L);


      if(EVI <= 0){
        evi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        evi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 0;
        evi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0 < EVI <= 0.25){
        evi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        evi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 128;
        evi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0.25 < EVI <= 0.5){
        evi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        evi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 255;
        evi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0.5 < EVI <= 0.75){
        evi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        evi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 255;
        evi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 0;
      }
      else{
        evi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        evi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 128;
        evi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 0;
      }
    }
  }

  cv::merge(evi_channels, evi_im);
}


void applySAVI(cv::Mat rgb_im, const cv::Mat &ir_im, cv::Mat &savi_im)
{
  std::vector<cv::Mat> rgb_channels;
  std::vector<cv::Mat> savi_channels;

  cv::split(rgb_im, rgb_channels);
  cv::split(savi_im, savi_channels);

  float NIR, RED, BLUE, SAVI;
  float L = 2.0f;

  for(int i = 0; i < rgb_im.rows; ++i)
  {
    for(int j = 0; j < rgb_im.cols; ++j)
    {
      NIR = ir_im.at<uchar>(i, j);
      RED = rgb_channels[RED_CHANNEL_INDEX].at<uchar>(i, j);

      SAVI =  ((NIR - RED) / (NIR + RED  + L)) *(1 + L);


      if(SAVI <= 0){
        savi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        savi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 0;
        savi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0 < SAVI <= 0.25){
        savi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        savi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 128;
        savi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0.25 < SAVI <= 0.5){
        savi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        savi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 255;
        savi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0.5 < SAVI <= 0.75){
        savi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        savi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 255;
        savi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 0;
      }
      else{
        savi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        savi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 128;
        savi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 0;
      }
    }
  }

  cv::merge(savi_channels, savi_im);
}

cv::Mat applyExG(const cv::Mat &image)
{
    double Rn, Gn, Bn;
    double r, g, b;

    cv::Mat output_image(image.rows, image.cols, CV_8U);

    std::vector<cv::Mat> rgb_channels;

    cv::split(image, rgb_channels);


    for(int32_t i = 0; i < image.rows; ++i){
        for(int32_t j = 0; j < image.cols; ++j){

            Rn = (double)rgb_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) / MAX_PIXEL_VALUE;  // Normalize RGB channels of image to range[0 1]
            Gn = (double)rgb_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) / MAX_PIXEL_VALUE;
            Bn = (double)rgb_channels[BLUE_CHANNEL_INDEX].at<uchar>(i, j) / MAX_PIXEL_VALUE;

            r = Rn/(Rn+Gn+Bn);
            g = Gn/(Rn+Gn+Bn);
            b = Bn/(Rn+Gn+Bn);

            double ExG = 2*g - r - b;

            double formula_value = ExG;

            if(formula_value >= MAX_INDEX_THRESHOLD)
                output_image.at<uchar>(i,j) = MAX_PIXEL_VALUE;
            else if(formula_value <= MIN_INDEX_THRESHOLD)
                output_image.at<uchar>(i,j) = MIN_PIXEL_VALUE;
            else
                output_image.at<uchar>(i,j) = formula_value * MAX_PIXEL_VALUE;

        }
    }

    return output_image;
}

void analyzeSquare(cv::Mat &image, const cv::Point2f &initPoint)
{
  const cv::Scalar RED(0, 0, 255);
  const cv::Scalar ORANGE(0, 128, 255);
  const cv::Scalar YELLOW(0, 255, 255);
  const cv::Scalar GREEN(0, 255, 0);
  const cv::Scalar DARK_GREEN(0, 128, 0);
  const cv::Scalar BLACK(0, 0, 0);

  int32_t badPixelsCount = 0;
  int32_t goodPixelsCount = 0;
  int32_t blackPixelsCount = 0;

  for(int32_t i = initPoint.x; ((i < initPoint.x+SQUARE_SIZE) && (i < image.rows)); ++i)
  {
    for(int32_t j = initPoint.y; ((j < initPoint.y+SQUARE_SIZE) && (j < image.cols)); ++j)
    {
      cv::Vec3b color = image.at<cv::Vec3b>(cv::Point(i,j));

      if(((color[0] == RED[0]) && (color[1] == RED[1]) && (color[2]  == RED[2])) || 
         ((color[0] == ORANGE[0]) && (color[1]  == ORANGE[1]) && (color[2] == ORANGE[2])) || 
         ((color[0] == YELLOW[0]) && (color[1]  == YELLOW[1]) && (color[2] == YELLOW[2])))
      {
        badPixelsCount++;    
      }
      else if(((color[0] == GREEN[0]) && (color[1]  == GREEN[1]) && (color[2] == GREEN[2])) ||
              ((color[0] == DARK_GREEN[0]) && (color[1]  == DARK_GREEN[1]) && (color[2] == DARK_GREEN[2])))
      {
        goodPixelsCount++;
      }
      else if((color[0] == BLACK[0]) && (color[1] == BLACK[1]) && (color[2] == BLACK[2]))
      {
        blackPixelsCount++;
      }
      else{}      
    }
  }

  std::cout << "Point: " << initPoint << " BAD PIXELS: " << badPixelsCount << " GOOD PIXELS: " << goodPixelsCount << " BLACK PIXELS: " << blackPixelsCount << std::endl;

  if((badPixelsCount > goodPixelsCount) && (badPixelsCount > blackPixelsCount))
  {
    cv::Point2f center(initPoint.x+(SQUARE_SIZE/2), initPoint.y+(SQUARE_SIZE/2));
    cv::circle(image, center, 10, cv::Scalar(255, 255, 255), 2);
  }
}

cv::Mat analyzeImage(const cv::Mat &image)
{
  cv::Mat gridImage = image.clone();

  // Creates grid on image
  // for(int32_t i = 0; i < image.rows; i+=SQUARE_SIZE)
  // {
  //   cv::line(gridImage, cv::Point2f(0, i), cv::Point2f(image.cols-1, i), 
  //           cv::Scalar(255,255,255), 2);

  //   for(int32_t j = i; j < image.cols; j+=SQUARE_SIZE)
  //   {
  //     cv::line(gridImage, cv::Point2f(j, 0), cv::Point2f(j, image.rows), 
  //             cv::Scalar(255,255,255), 2);
  //   }
  // }

  // Analyze crops health by image chunks
  for(int32_t i = 0; i <= gridImage.rows-1; i+=SQUARE_SIZE)
  {
    for(int32_t j = 0; j <= gridImage.cols-1; j+=SQUARE_SIZE)
    {
      analyzeSquare(gridImage, cv::Point2f(i, j));
    }
  }

  return gridImage;
}

void createVegetationIndexImages(const std::vector<WAYPOINT_DATA> &waypointsData)
{

  for(int32_t i = 0; i < waypointsData.size(); ++i)
  {
    cv::Mat eviImage(waypointsData[i].rgbImg.size(), CV_8UC3, cv::Scalar(0,0,0));
    cv::Mat saviImage(waypointsData[i].rgbImg.size(), CV_8UC3, cv::Scalar(0,0,0));
    const std::string saviName = "savi_" + waypointsData[i].rgbImgName;
    const std::string eviName = "evi_" + waypointsData[i].rgbImgName;
    cv::Mat otsuImage;

    cv::Mat exgImage = applyExG(waypointsData[i].rgbImg);


    // Convert green index in binary values -> mask
    (void)cv::threshold(exgImage, otsuImage, OTSU_MIN_THRESHOLD, 
                        OTSU_MAX_THRESHOLD, cv::THRESH_OTSU);

    // Convert mask to RGB format
    cv::cvtColor(otsuImage, otsuImage, cv::COLOR_GRAY2BGR);

    // Apply vegetation indexes
    applySAVI(waypointsData[i].rgbImg, waypointsData[i].irImg, saviImage);
    applyEVI(waypointsData[i].rgbImg, waypointsData[i].irImg, eviImage);

    // Remove pixels from index which dont have vegetation
    cv::bitwise_and(saviImage, otsuImage, saviImage);
    cv::bitwise_and(eviImage, otsuImage, eviImage);

    // Add critical points where the crop health is very poor
    const cv::Mat saviAnalyzed = analyzeImage(saviImage);
    const cv::Mat eviAnalyzed = analyzeImage(eviImage);

    cv::imshow("SAVI", saviAnalyzed);
    cv::imshow("EVI", eviAnalyzed);

    cv::waitKey(0);

    cv::Mat hSavi, hEvi;

    cv::hconcat(waypointsData[i].rgbImg, saviAnalyzed, hSavi);
    cv::hconcat(waypointsData[i].rgbImg, eviAnalyzed, hEvi);

    cv::imwrite(saviName, hSavi);
    cv::imwrite(eviName, hEvi);
  }

}



int32_t main (int32_t argc, char * argv[]) 
{
  cv::Mat rgbImage, stitchedIR;

  // Check arguments before processing
  checkArgs(argc, argv);

  // Initialize list of waipoints metadata
  const std::vector<WAYPOINT_DATA> waypointsData = initWaypointsData(argv[1]);

  createVegetationIndexImages(waypointsData);

  // Wait to press a key
  cv::waitKey(ANY_KEY);
  
  return 0;
}
