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
#include "waypoint_data.h"

constexpr int8_t ARGC = 2;
constexpr int8_t ANY_KEY = 0;
constexpr int8_t NUM_COLUMNS = 11;
constexpr int8_t RGB_NAME_INDEX = 0;
constexpr int8_t IR_NAME_INDEX = 1;
constexpr int8_t X_INDEX = 2;
constexpr int8_t Y_INDEX = 3;
constexpr int8_t Z_INDEX = 4;
constexpr int8_t LAT_INDEX = 5;
constexpr int8_t LON_INDEX = 6;
constexpr int8_t ALT_INDEX = 7;
constexpr int8_t ROLL_INDEX = 8;
constexpr int8_t PITCH_INDEX = 9;
constexpr int8_t YAW_INDEX = 10;
const std::string HELP_ARG = "--help";
const std::string HELP_FLAG = "-h";
const std::string SLASH_STR = "/";
const std::string CSV_FILENAME = "metadata.csv";
constexpr char COMMA_DELIM = ',';
constexpr int32_t RED_CHANNEL_INDEX = 2;
constexpr int32_t GREEN_CHANNEL_INDEX = 1;
constexpr int32_t BLUE_CHANNEL_INDEX = 0;
constexpr int32_t OTSU_MIN_THRESHOLD = 0;
constexpr int32_t OTSU_MAX_THRESHOLD = 255;
constexpr int32_t MIN_PIXEL_VALUE = 0;
constexpr int32_t MAX_PIXEL_VALUE = 255;
constexpr int32_t MIN_INDEX_THRESHOLD = 0;
constexpr int32_t MAX_INDEX_THRESHOLD = 1;
constexpr int32_t SQUARE_SIZE = 80;

//  Colors used for indexes
const cv::Scalar RED(0, 0, 255);
const cv::Scalar ORANGE(0, 128, 255);
const cv::Scalar YELLOW(0, 255, 255);
const cv::Scalar GREEN(0, 255, 0);
const cv::Scalar DARK_GREEN(0, 128, 0);
const cv::Scalar BLACK(0, 0, 0);



static std::vector<WAYPOINT_DATA> m_waypointsData;



void printUsage(const bool help=false)
{
  if(true == help)
  {
    std::cout << "usage: post_mission_processing [mission_dir]" << std::endl;
    std::cout << std::endl << "Note: mission_dir is the path to the auto-generated directory during the mission." 
    "It must contain a CSV file named 'metadata.csv' with columns: "
          "rgb_img_name,ir_img_name,x,y,z,latitude,longitude,altitude,roll,pitch,yaw." << std::endl;
  }
  else
  {
    std::cout << "usage: periodic_mission_post_processing [-h/--help] for more detailed usage." << std::endl;
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


void initWaypointsData(const char *dirPath)
{
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

      const cv::Mat rgbImage = cv::imread(rgbImagePath, cv::IMREAD_COLOR);
      const cv::Mat irImage = cv::imread(irImagePath, cv::IMREAD_GRAYSCALE);

      if((false == rgbImage.empty()) && (false == irImage.empty()))
      {
        const double x = std::atof(rowValues[X_INDEX].c_str());
        const double y = std::atof(rowValues[Y_INDEX].c_str());
        const double z = std::atof(rowValues[Z_INDEX].c_str());
        const double lat = std::atof(rowValues[LAT_INDEX].c_str());
        const double lon = std::atof(rowValues[LON_INDEX].c_str());
        const double alt = std::atof(rowValues[ALT_INDEX].c_str());
        const double roll = std::atof(rowValues[ROLL_INDEX].c_str());
        const double pitch = std::atof(rowValues[PITCH_INDEX].c_str());
        const double yaw = std::atof(rowValues[YAW_INDEX].c_str());

        const WAYPOINT_DATA auxStruct(
          rgbImage, rowValues[RGB_NAME_INDEX], 
          irImage, rowValues[IR_NAME_INDEX], 
          x, y, z, 
          lat, lon,alt, 
          roll, pitch, yaw);
        
        m_waypointsData.push_back(auxStruct);
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
}

void applyNDVI(cv::Mat rgb_im, cv::Mat ir_im, cv::Mat &ndvi_im)
{
  std::vector<cv::Mat> rgb_channels;
  std::vector<cv::Mat> ndvi_channels;

  cv::split(rgb_im, rgb_channels);
  cv::split(ndvi_im, ndvi_channels);

  double NIR, RED, NDVI;

  for(int i = 0; i < rgb_im.rows; ++i){
    for(int j = 0; j < rgb_im.cols; ++j){
      NIR = ir_im.at<uchar>(i, j);
      RED = rgb_channels[RED_CHANNEL_INDEX].at<uchar>(i, j);


      NDVI = (NIR - RED) / (NIR + RED);


      if(NDVI <= 0)
      {
        ndvi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        ndvi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 0;
        ndvi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0 < NDVI <= 0.25)
      {
        ndvi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        ndvi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 128;
        ndvi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0.25 < NDVI <= 0.5)
      {
        ndvi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        ndvi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 255;
        ndvi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0.5 < NDVI <= 0.75)
      {
        ndvi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        ndvi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 255;
        ndvi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 0;
      }
      else
      {
        ndvi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        ndvi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 128;
        ndvi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 0;
      }
    }
  }

  cv::merge(ndvi_channels, ndvi_im);
}


void applyEVI(cv::Mat rgb_im, cv::Mat ir_im, cv::Mat &evi_im)
{
  std::vector<cv::Mat> rgb_channels;
  std::vector<cv::Mat> evi_channels;

  cv::split(rgb_im, rgb_channels);
  cv::split(evi_im, evi_channels);

  std::cout << rgb_im.size() << std::endl;

  float NIR, RED, BLUE, EVI;
  float G = 4.5f;
  float C1 = 6.0f;
  float C2 = 4.5f;
  float L = 1.0f;

  for(int i = 0; i < rgb_im.rows; ++i)
  {
    for(int j = 0; j < rgb_im.cols; ++j)
    {
      NIR = ir_im.at<uchar>(i, j);
      RED = rgb_channels[RED_CHANNEL_INDEX].at<uchar>(i, j);
      BLUE = rgb_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j);

      EVI =  G * (NIR -RED) / (NIR + C1 * RED - C2 * BLUE + L);


      if(EVI <= 0)
      {
        evi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        evi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 0;
        evi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0 < EVI <= 0.25)
      {
        evi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        evi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 128;
        evi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0.25 < EVI <= 0.5)
      {
        evi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        evi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 255;
        evi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0.5 < EVI <= 0.75)
      {
        evi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        evi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 255;
        evi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 0;
      }
      else
      {
        evi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        evi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 128;
        evi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 0;
      }
    }
  }

  cv::merge(evi_channels, evi_im);
}


void applySAVI(cv::Mat rgb_im, cv::Mat ir_im, cv::Mat &savi_im)
{
  std::vector<cv::Mat> rgb_channels;
  std::vector<cv::Mat> savi_channels;

  cv::split(rgb_im, rgb_channels);
  cv::split(savi_im, savi_channels);

  float NIR, RED, BLUE, SAVI;
  float L = 2.0f;

  for(int i = 0; i < savi_im.rows; ++i)
  {
    for(int j = 0; j < savi_im.cols; ++j)
    {
      NIR = ir_im.at<uchar>(i, j);
      RED = rgb_channels[RED_CHANNEL_INDEX].at<uchar>(i, j);

      SAVI =  ((NIR - RED) / (NIR + RED  + L)) *(1 + L);


      if(SAVI <= 0)
      {
        savi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        savi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 0;
        savi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0 < SAVI <= 0.25)
      {
        savi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        savi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 128;
        savi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0.25 < SAVI <= 0.5)
      {
        savi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        savi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 255;
        savi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 255;
      }
      else if(0.5 < SAVI <= 0.75)
      {
        savi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        savi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 255;
        savi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 0;
      }
      else
      {
        savi_channels[BLUE_CHANNEL_INDEX].at<uchar>(i,j) = 0;
        savi_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) = 128;
        savi_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) = 0;
      }
    }
  }

  cv::merge(savi_channels, savi_im);
}


cv::Mat applyExG(cv::Mat image)
{
    double Rn, Gn, Bn;
    double r, g, b;

    cv::Mat output_image(image.rows, image.cols, CV_8U);

    std::vector<cv::Mat> rgb_channels;

    cv::split(image, rgb_channels);


    for(int32_t i = 0; i < output_image.rows; ++i)
    {
        for(int32_t j = 0; j < output_image.cols; ++j)
        {

            Rn = (double)rgb_channels[RED_CHANNEL_INDEX].at<uchar>(i, j) / MAX_PIXEL_VALUE;  // Normalize RGB channels of image to range[0 1]
            Gn = (double)rgb_channels[GREEN_CHANNEL_INDEX].at<uchar>(i, j) / MAX_PIXEL_VALUE;
            Bn = (double)rgb_channels[BLUE_CHANNEL_INDEX].at<uchar>(i, j) / MAX_PIXEL_VALUE;

            r = Rn/(Rn+Gn+Bn);
            g = Gn/(Rn+Gn+Bn);
            b = Bn/(Rn+Gn+Bn);

            double ExG = 2*g - r - b;

            double formula_value = ExG;

            if(formula_value >= MAX_INDEX_THRESHOLD)
            {
              output_image.at<uchar>(i,j) = MAX_PIXEL_VALUE;
            }
            else if(formula_value <= MIN_INDEX_THRESHOLD)
            {
              output_image.at<uchar>(i,j) = MIN_PIXEL_VALUE;
            }
            else
            {
              output_image.at<uchar>(i,j) = formula_value * MAX_PIXEL_VALUE;
            }
        }
    }

    return output_image;
}


void analyzeSquare(const cv::Mat &image, const cv::Point2f &initPoint, const int32_t index)
{
  int32_t badPixelsCount = 0;
  int32_t goodPixelsCount = 0;
  int32_t blackPixelsCount = 0;

  for(int32_t i = initPoint.x; ((i < initPoint.x+SQUARE_SIZE) && (i < image.rows)); ++i)
  {
    for(int32_t j = initPoint.y; ((j < initPoint.y+SQUARE_SIZE) && (j < image.cols)); ++j)
    {
      const cv::Vec3b color = image.at<cv::Vec3b>(i,j);

      if(((color[BLUE_CHANNEL_INDEX] == RED[BLUE_CHANNEL_INDEX]) && 
            (color[GREEN_CHANNEL_INDEX] == RED[GREEN_CHANNEL_INDEX]) && 
            (color[RED_CHANNEL_INDEX]  == RED[RED_CHANNEL_INDEX])) || 
         ((color[BLUE_CHANNEL_INDEX] == ORANGE[BLUE_CHANNEL_INDEX]) && 
            (color[GREEN_CHANNEL_INDEX]  == ORANGE[GREEN_CHANNEL_INDEX]) && 
            (color[RED_CHANNEL_INDEX] == ORANGE[RED_CHANNEL_INDEX])))
      {
        badPixelsCount++;    
      }
      else if(((color[BLUE_CHANNEL_INDEX] == GREEN[BLUE_CHANNEL_INDEX]) && 
                (color[GREEN_CHANNEL_INDEX]  == GREEN[GREEN_CHANNEL_INDEX]) && 
                (color[RED_CHANNEL_INDEX] == GREEN[RED_CHANNEL_INDEX])) ||
              ((color[BLUE_CHANNEL_INDEX] == DARK_GREEN[BLUE_CHANNEL_INDEX]) && 
                (color[GREEN_CHANNEL_INDEX]  == DARK_GREEN[GREEN_CHANNEL_INDEX]) && 
                (color[RED_CHANNEL_INDEX] == DARK_GREEN[RED_CHANNEL_INDEX])) || 
              ((color[BLUE_CHANNEL_INDEX] == YELLOW[BLUE_CHANNEL_INDEX]) && 
                (color[GREEN_CHANNEL_INDEX]  == YELLOW[GREEN_CHANNEL_INDEX]) && 
              (color[RED_CHANNEL_INDEX] == YELLOW[RED_CHANNEL_INDEX])))
      {
        goodPixelsCount++;
      }
      else if((color[BLUE_CHANNEL_INDEX] == BLACK[BLUE_CHANNEL_INDEX]) && 
              (color[GREEN_CHANNEL_INDEX] == BLACK[GREEN_CHANNEL_INDEX]) && 
              (color[RED_CHANNEL_INDEX] == BLACK[RED_CHANNEL_INDEX]))
      {
        blackPixelsCount++;
      }
      else{}      
    }
  }

  // std::cout << "CENTER: " << initPoint << " GOOD PIXELS:" << goodPixelsCount << 
  //              " BAD PIXELS: " << badPixelsCount << " BLACK PIXELS: " << blackPixelsCount << 
  //              " CONDITION: " << ((badPixelsCount > goodPixelsCount) && (badPixelsCount > blackPixelsCount)) << std::endl;

  if((badPixelsCount > goodPixelsCount) && ((badPixelsCount + goodPixelsCount) > blackPixelsCount))
  {
    cv::Point2f center(initPoint.y+(SQUARE_SIZE/2), initPoint.x+(SQUARE_SIZE/2));

    m_waypointsData[index].criticalPoints.push_back(center);
  }
}


cv::Mat createGrid(const cv::Mat &image)
{
  cv::Mat gridImage = image.clone();

  // Creates grid on image
  for(int32_t i = 0; i < gridImage.rows; i+=SQUARE_SIZE)
  {
    cv::line(gridImage, cv::Point2f(0, i), cv::Point2f(image.cols-1, i), 
            cv::Scalar(255,255,255), 2);

    for(int32_t j = i; j < image.cols; j+=SQUARE_SIZE)
    {
      cv::line(gridImage, cv::Point2f(j, 0), cv::Point2f(j, image.rows), 
              cv::Scalar(255,255,255), 2);
    }
  }

  return gridImage;
}


cv::Mat analyzeImage(const cv::Mat &image, const int32_t index)
{

  const cv::Mat gridImage = createGrid(image);

  // Analyze crops health by image chunks
  for(int32_t i = 0; i < gridImage.rows; i+=SQUARE_SIZE)
  {
    for(int32_t j = 0; j < gridImage.cols; j+=SQUARE_SIZE)
    {
      analyzeSquare(gridImage, cv::Point2f(i, j), index);
    }
  }

  return gridImage;
}

void paintCenterPoint(WAYPOINT_DATA &wpData)
{
    static const cv::Scalar DOT_COLOR(255,255,255);
  double cx = wpData.rgbImg.cols / 2.0; 
  double cy = wpData.rgbImg.rows / 2.0;
  static constexpr double FOCAL_LENGTH_X = 425.94;
  static constexpr double FOCAL_LENGTH_Y = 425.94;
  static constexpr double EARTH_RADIUS = 6.378e6;
  //  Offset of GPS from camera
  static constexpr double GPS_OFFSET_Y = 0.2;
  static constexpr double GPS_OFFSET_Z = 0.15;
  //  Coordinates offset
  static constexpr double LAT_OFFSET = 0.000210;
  static constexpr double LON_OFFSET = 0.000168;
  const char * const LAT_FORMAT = "LAT: %.6f";
  const char * const LON_FORMAT = "LON: %.6f";
  const double MISSION_ALT = -wpData.z + GPS_OFFSET_Z;


  //  Distance in metres
  double dx = ((cx - cx) * MISSION_ALT) / FOCAL_LENGTH_X;
  double dy = (((cy - cy) * MISSION_ALT) / FOCAL_LENGTH_Y) + GPS_OFFSET_Y;

  std::cout << "DX: " << dx << " DY: " << dy << std::endl;

  // std::cout << "DX: " << dx << " DY: " << dy << std::endl;

  // std::cout << "LAT: " << wpData.lat << " LON: " << wpData.lon << std::endl;

  double critPointLat = (wpData.lat + (dy / EARTH_RADIUS) * (180.0 / M_PI)) /*+ LAT_OFFSET*/;
  double critPointLon = (wpData.lon + (dx / EARTH_RADIUS) * 
                        (180.0 / M_PI) / cos(wpData.lat * M_PI/180.0)) /*+ LON_OFFSET*/;

  char latBuff[40], lonBuff[40];

  std::snprintf(latBuff, 40, LAT_FORMAT, critPointLat);
  std::snprintf(lonBuff, 40, LON_FORMAT, critPointLon);


  const std::string latPointStr = std::string(latBuff);
  const std::string lonPointStr = std::string(lonBuff);

  //  Paint circle and text
  cv::circle(wpData.rgbImg, cv::Point(cx,cy), 3, DOT_COLOR, 2);


  cv::Point2f latTextPoint, lonTextPoint;

  latTextPoint = cv::Point2f(cx - 30, cy - 20);
  lonTextPoint = cv::Point2f(cx - 30, cy - 10);


  cv::putText(wpData.rgbImg, latPointStr, latTextPoint, cv::FONT_HERSHEY_SIMPLEX, 
              0.25, DOT_COLOR, 1);
  cv::putText(wpData.rgbImg, lonPointStr, lonTextPoint, cv::FONT_HERSHEY_SIMPLEX, 
              0.25, DOT_COLOR, 1);
}


void paintCriticalPointsCoordinates(WAYPOINT_DATA &wpData)
{
  static const cv::Scalar DOT_COLOR(255,255,255);
  double cx = wpData.rgbImg.cols / 2.0; 
  double cy = wpData.rgbImg.rows / 2.0;
  static constexpr double FOCAL_LENGTH_X = 425.94;
  static constexpr double FOCAL_LENGTH_Y = 425.94;
  static constexpr double EARTH_RADIUS = 6.378e6;
  //  Offset of GPS from camera
  static constexpr double GPS_OFFSET_Y = 0.20;
  static constexpr double GPS_OFFSET_Z = 0.15;
  //  Coordinates offset
  static constexpr double LAT_OFFSET = 0.000210;
  static constexpr double LON_OFFSET = 0.000168;
  const char * const LAT_FORMAT = "LAT: %.6f";
  const char * const LON_FORMAT = "LON: %.6f";
  const double MISSION_ALT = -wpData.z + GPS_OFFSET_Z;
  


  for(auto const &critPoint : wpData.criticalPoints)
  { 
    //  Distance in metres
    double dx = ((critPoint.x - cx) * MISSION_ALT) / FOCAL_LENGTH_X;
    double dy = (((critPoint.y - cy) * MISSION_ALT) / FOCAL_LENGTH_Y) /*+ GPS_OFFSET_Y*/;

    const double angle = wpData.yaw;

    double dxh = -dy*cos(angle) - dx*sin(angle);
    double dyh = -dy*sin(angle) + dx*cos(angle);

    std::cout << "DX: " << dx << " DY: " << dy << std::endl;

    std::cout << "LAT: " << wpData.lat << " LON: " << wpData.lon << std::endl;

    double critPointLat = (wpData.lat + (dxh / EARTH_RADIUS) * (180.0 / M_PI)) /*+ LAT_OFFSET*/;
    double critPointLon = (wpData.lon + (dyh / EARTH_RADIUS) * 
                          (180.0 / M_PI) / cos(wpData.lat * M_PI/180.0)) /*+ LON_OFFSET*/;


    std::cout << " NEW LAT: " << critPointLat << " NEW LON: " << critPointLon << std::endl;

    char latBuff[40], lonBuff[40];

    std::snprintf(latBuff, 40, LAT_FORMAT, critPointLat);
    std::snprintf(lonBuff, 40, LON_FORMAT, critPointLon);


    const std::string latPointStr = std::string(latBuff);
    const std::string lonPointStr = std::string(lonBuff);

    //  Paint circle and text
    cv::circle(wpData.rgbImg, critPoint, 3, DOT_COLOR, 2);


    cv::Point2f latTextPoint, lonTextPoint;

    latTextPoint = cv::Point2f(critPoint.x - 30, critPoint.y - 20);
    lonTextPoint = cv::Point2f(critPoint.x - 30, critPoint.y - 10);


    cv::putText(wpData.rgbImg, latPointStr, latTextPoint, cv::FONT_HERSHEY_SIMPLEX, 
                0.25, DOT_COLOR, 1);
    cv::putText(wpData.rgbImg, lonPointStr, lonTextPoint, cv::FONT_HERSHEY_SIMPLEX, 
                0.25, DOT_COLOR, 1);
  }
}


void createVegetationIndexImages()
{

  for(int32_t i = 0; i < m_waypointsData.size(); ++i)
  {
    cv::Mat ndviImage(m_waypointsData[i].rgbImg.size(), CV_8UC3, cv::Scalar(0,0,0));
    cv::Mat eviImage(m_waypointsData[i].rgbImg.size(), CV_8UC3, cv::Scalar(0,0,0));
    cv::Mat saviImage(m_waypointsData[i].rgbImg.size(), CV_8UC3, cv::Scalar(0,0,0));
    const std::string saviName = "savi_" + m_waypointsData[i].rgbImgName;
    const std::string eviName = "evi_" + m_waypointsData[i].rgbImgName;
    cv::Mat otsuImage;

    cv::Mat exgImage = applyExG(m_waypointsData[i].rgbImg);

    // Convert green index in binary values -> mask
    (void)cv::threshold(exgImage, otsuImage, OTSU_MIN_THRESHOLD, 
                        OTSU_MAX_THRESHOLD, cv::THRESH_OTSU);


    //  Improve mask
    const cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9,9));
    cv::morphologyEx(otsuImage, otsuImage, cv::MORPH_CLOSE, element, cv::Point(-1,-1), 3);
    cv::morphologyEx(otsuImage, otsuImage, cv::MORPH_OPEN, element, cv::Point(-1,-1));


    // cv::imshow("DILATED MASK", otsuImage);
    // cv::waitKey(0);

    // Convert mask to RGB format
    cv::cvtColor(otsuImage, otsuImage, cv::COLOR_GRAY2BGR);

    // Apply vegetation index
    applyNDVI(m_waypointsData[i].rgbImg, m_waypointsData[i].irImg, ndviImage);
    applySAVI(m_waypointsData[i].rgbImg, m_waypointsData[i].irImg, saviImage);
    applyEVI(m_waypointsData[i].rgbImg, m_waypointsData[i].irImg, eviImage);


    // Remove pixels from index which dont have vegetation
    cv::bitwise_and(ndviImage, otsuImage, ndviImage);
    cv::bitwise_and(saviImage, otsuImage, saviImage);
    cv::bitwise_and(eviImage, otsuImage, eviImage);


    // Add critical points where the crop health is very poor
    // const cv::Mat saviAnalyzed = analyzeImage(saviImage, i);
    const cv::Mat eviAnalyzed = analyzeImage(eviImage, i);

    //  Paint critical points in image with GPS coordinates
    paintCriticalPointsCoordinates(m_waypointsData[i]);

    //  Paint center point of image and coordinates to check GPS estimation error (optional)
    // paintCenterPoint(m_waypointsData[i]);

    const cv::Mat rgbWGrid = createGrid(m_waypointsData[i].rgbImg);

    cv::Mat hSavi, hEvi;

    // cv::hconcat(m_waypointsData[i].rgbImg, saviAnalyzed, hSavi);
    cv::hconcat(rgbWGrid, eviAnalyzed, hEvi);

    cv::imshow("IMAGE ANALYZED", hEvi);

    cv::waitKey(0);

    // cv::imwrite(saviName, hSavi);
    cv::imwrite(eviName, hEvi);
  }

}


int32_t main (int32_t argc, char * argv[]) 
{
  std::cout << std::setprecision(10) << std::fixed;

  // Check arguments before processing
  checkArgs(argc, argv);

  // Initialize list of waipoints metadata
  initWaypointsData(argv[1]);

  createVegetationIndexImages();

  // Wait to press a key
  cv::waitKey(ANY_KEY);
  
  return 0;
}
