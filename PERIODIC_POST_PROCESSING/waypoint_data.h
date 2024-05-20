#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"

constexpr int32_t MAX_FEATURES = 9000;
constexpr float GOOD_MATCH_PERCENT = 0.15f;
const cv::Size IMAGE_SIZE(840, 480);


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
  double lat;
  double lon;
  double alt;
  double roll;
  double pitch;
  double yaw;
  std::vector<cv::Point2f> criticalPoints;

  WAYPOINT_DATA(const cv::Mat &rgbImg_,
                const std::string &rgbImgName_,
                const cv::Mat &irImg_,
                const std::string &irImgName_,
                const double x_,
                const double y_,
                const double z_,
                const double lat_,
                const double lon_,
                const double alt_,
                const double roll_,
                const double pitch_,
                const double yaw_)
  {
    rgbImg = rgbImg_.clone();
    rgbImgName = rgbImgName_;
    irImg = irImg_.clone();
    irImgName = irImgName_;
    x = x_;
    y = y_;
    z = z_;
    lat = lat_;
    lon = lon_;
    alt = alt_;
    roll = roll_;
    pitch = pitch_;
    yaw = yaw_;
    criticalPoints = {};


    printf("ALIGNING IMAGES... %s AND %s\n", rgbImgName.c_str(), irImgName.c_str()); 
    irImg = registerIrImage(rgbImg, irImg);

    cv::resize(rgbImg, rgbImg, IMAGE_SIZE, cv::INTER_LINEAR);
    cv::resize(irImg, irImg, IMAGE_SIZE, cv::INTER_LINEAR);


    cropImages();
  }

    void cropImages()
  {
    rgbImg = rgbImg(cv::Range(0, rgbImg.rows), cv::Range(200, rgbImg.cols));
    irImg = irImg(cv::Range(0, irImg.rows), cv::Range(200, irImg.cols));
  }


};