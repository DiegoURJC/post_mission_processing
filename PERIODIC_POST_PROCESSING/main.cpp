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

        // Rotate images with registered yaw
    // printf("ROTATING IMAGES... %s AND %s\n", rgbImgName.c_str(), irImgName.c_str());  
    // rotateImages();


    // cv::imshow("ROTATED RGB", rgbImg);
    // cv::imshow("ROTATED IR", irImg);

    // cv::waitKey(0);
  }

  void rotateImages()
  {
    // Get images center
    const cv::Point2f rgbCenter(rgbImg.cols / 2.0, rgbImg.rows / 2.0);
    const cv::Point2f irCenter(irImg.cols / 2.0, irImg.rows / 2.0);

    const double angle = (yaw*180.0) / M_PI;

    // get rotation matrix for rotating the image around its center in pixel coordinates
    cv::Mat rgbRot = cv::getRotationMatrix2D(rgbCenter, -angle, 1.0);
    cv::Mat irRot = cv::getRotationMatrix2D(irCenter, -angle, 1.0);

    cv::warpAffine(rgbImg, rgbImg, rgbRot, rgbImg.size());
    cv::warpAffine(irImg, irImg, irRot, irImg.size());

    cv::imshow("ROTATED RGB", rgbImg);
    cv::imshow("ROTATED IR", irImg);
  }

  void cropImages()
  {
    // cv::Mat prevRgb = rgbImg;
    // cv::Mat prevIr = irImg;

    // cv::Mat  irConc, rgbConc;

    rgbImg = rgbImg(cv::Range(0, rgbImg.rows), cv::Range(200, rgbImg.cols));
    irImg = irImg(cv::Range(0, irImg.rows), cv::Range(200, irImg.cols));

    // cv::hconcat(prevRgb, rgbImg, rgbConc);
    // cv::hconcat(prevIr, irImg, irConc);

    // cv::imshow("IR", irConc);
    // cv::imshow("RGB", rgbConc);

    // cv::waitKey(ANY_KEY);

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

void stitchImages(const std::vector<WAYPOINT_DATA> &waypointsData, cv::Mat &rgb, cv::Mat &ir)
{
  std::vector<cv::Mat> rgbImages, irImages;

  for(auto const wpData : waypointsData)
  {
    rgbImages.push_back(wpData.rgbImg);
    irImages.push_back(wpData.irImg);
  }

  cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);

  std::cout << rgbImages.size() << " " << irImages.size() << std::endl;

  const cv::Stitcher::Status rgbStatus = stitcher->stitch(rgbImages, rgb);
  const cv::Stitcher::Status irStatus = stitcher->stitch(irImages, ir);

  std::cout << rgbStatus << " " << irStatus << std::endl;
}

cv::Mat applyNDVI(cv::Mat rgbImg, const cv::Mat &irImg)
{
  cv::Mat ndviImg(rgbImg.size(), CV_8UC3, cv::Scalar(0,0,0));
  std::vector<cv::Mat> rgbChannels;
  std::vector<cv::Mat> ndviChannels;

  cv::split(rgbImg, rgbChannels);
  cv::split(ndviImg, ndviChannels);

  double NIR, RED, NDVI;

  for(int i = 0; i < rgbImg.rows; ++i)
  {
    for(int j = 0; j < rgbImg.cols; ++j)
    {
      NIR = irImg.at<uchar>(i, j);
      RED = rgbChannels[RED_CHANNEL_INDEX].at<uchar>(i, j);


      NDVI = (NIR - RED) / (NIR + RED);

      if(NDVI <= 0)
      {
        ndviChannels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        ndviChannels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 0;
        ndviChannels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0 < NDVI <= 0.25)
      {
        ndviChannels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        ndviChannels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 128;
        ndviChannels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0.25 < NDVI <= 0.5)
      {
        ndviChannels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        ndviChannels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 255;
        ndviChannels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0.5 < NDVI <= 0.75)
      {
        ndviChannels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        ndviChannels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 255;
        ndviChannels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 0;
      }
      else
      {
        ndviChannels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        ndviChannels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 128;
        ndviChannels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 0;
      }
    }
  }

  cv::merge(ndviChannels, ndviImg);

  return ndviImg;
}

void applyGCI(cv::Mat rgb_im, const cv::Mat &ir_im, cv::Mat &gci_im)
{
  std::vector<cv::Mat> rgb_channels;
  std::vector<cv::Mat> gci_channels;

  cv::split(rgb_im, rgb_channels);
  cv::split(gci_im, gci_channels);

  float NIR, RED, BLUE, GREEN, GCI;

  for(int i = 0; i < rgb_im.rows; ++i){
    for(int j = 0; j < rgb_im.cols; ++j){
      NIR = ir_im.at<uchar>(i, j);
      RED = rgb_channels[RED_CHANNEL_INDEX].at<uchar>(i, j);
      BLUE = rgb_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j);
      GREEN = rgb_channels[GREEN_CHANNEL_INDEX].at<uchar>(i,j);

      GCI = (NIR / GREEN) - 1;


      if(GCI <= 0){
        gci_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        gci_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 0;
        gci_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0 < GCI <= 0.25){
        gci_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        gci_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 128;
        gci_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0.25 < GCI <= 0.5){
        gci_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        gci_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 255;
        gci_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0.5 < GCI <= 0.75){
        gci_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        gci_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 255;
        gci_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 0;
      }
      else{
        gci_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        gci_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 128;
        gci_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 0;
      }
    }
  }

  cv::merge(gci_channels, gci_im);
}

void applySIPI(cv::Mat rgb_im, const cv::Mat &ir_im, cv::Mat &sipi_im)
{
  std::vector<cv::Mat> rgb_channels;
  std::vector<cv::Mat> sipi_channels;

  cv::split(rgb_im, rgb_channels);
  cv::split(sipi_im, sipi_channels);

  float NIR, RED, BLUE, SIPI;

  for(int i = 0; i < rgb_im.rows; ++i){
    for(int j = 0; j < rgb_im.cols; ++j){
      NIR = ir_im.at<uchar>(i, j);
      RED = rgb_channels[RED_CHANNEL_INDEX].at<uchar>(i, j);
      BLUE = rgb_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j);

      SIPI = (NIR - BLUE) / (NIR - RED);


      if(SIPI <= 0){
        sipi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        sipi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 0;
        sipi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0 < SIPI <= 0.25){
        sipi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        sipi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 128;
        sipi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0.25 < SIPI <= 0.5){
        sipi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        sipi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 255;
        sipi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0.5 < SIPI <= 0.75){
        sipi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        sipi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 255;
        sipi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 0;
      }
      else{
        sipi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        sipi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 128;
        sipi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 0;
      }
    }
  }

  cv::merge(sipi_channels, sipi_im);
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



int32_t main (int32_t argc, char * argv[]) 
{
  cv::Mat rgbImage, stitchedIR;

  // Check arguments before processing
  checkArgs(argc, argv);

  // Initialize list of waipoints metadata
  const std::vector<WAYPOINT_DATA> waypointsData = initWaypointsData(argv[1]);

  std::vector<cv::Mat> saviImgs;

  // Stitch RGB images in one and IR images in another one
  stitchImages(waypointsData, rgbImage, stitchedIR);


  // Register stitched IR image with RGB image
  const cv::Mat irImage = registerIrImage(rgbImage, stitchedIR);

  cv::imshow("STITCHED RGB", rgbImage);
  cv::imshow("STITCHED IR", irImage);


  cv::Mat eviImage(rgbImage.size(), CV_8UC3, cv::Scalar(0,0,0));
  cv::Mat saviImage(rgbImage.size(), CV_8UC3, cv::Scalar(0,0,0));

  std::cout << "RGB: " << rgbImage.rows << " " << rgbImage.cols << std::endl; 
  std::cout << "IR: " << irImage.rows << " " << irImage.cols << std::endl; 

  const cv::Mat ndviImage = applyNDVI(rgbImage, irImage);
  applyEVI(rgbImage, irImage, eviImage);
  applySAVI(rgbImage, irImage, saviImage);


  cv::imshow("NDVI IMAGE", ndviImage);
  cv::imshow("EVI IMAGE", eviImage);
  cv::imshow("SAVI IMAGE", saviImage);

  // Wait to press a key
  cv::waitKey(ANY_KEY);
  
  return 0;
}
