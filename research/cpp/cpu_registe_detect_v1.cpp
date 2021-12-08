#include <iostream>
#include <tuple>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/cudafilters.hpp"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudalegacy.hpp>
#include <chrono>
#include <opencv2/xfeatures2d/cuda.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>

#define DESC_ORB 0
#define M_BF 0
#define M_FLANN 1

using namespace std;

using namespace cv;

using namespace cv::cuda;

cv::Rect2i operator*(cv::Rect2i &r, const double &scale)
{
    if (scale <= 1.0)
        return r;
    int height_ext = (int)r.height * (scale - 1.0);
    int width_ext = (int)r.width * (scale - 1.0);
    r.height += height_ext;
    r.width += width_ext;
    r.x -= width_ext / 2;
    r.y -= height_ext / 2;
    return r;
}

cv::Rect2i operator*=(cv::Rect2i &r, const double &scale)
{
    if (scale <= 1.0)
        return r;
    int height_ext = (int)r.height * (scale - 1.0);
    int width_ext = (int)r.width * (scale - 1.0);
    r.height += height_ext;
    r.width += width_ext;
    r.x -= width_ext / 2;
    r.y -= height_ext / 2;
    return r;
}

bool operator<(cv::Rect2i &r1, cv::Rect2i &r2)
{
    return (r1 & r2) == r1;
}


const 

// @h_object_image: gray base image
// @h_scene_image: gray target image with possiable flaws
// @function: warp base image to fit into the target image; as there may be
//            black boundary after warp, target image is masked; futher, as
//            the mask's boundary is not smooth, base image also is masked;
void image_registration(Mat &h_object_image, Mat &h_scene_image, Mat &h_mask, int desc_type = DESC_ORB, int m_type = M_BF)
{
    vector<KeyPoint> h_keypoints_object, h_keypoints_scene; // CPU key points
    Mat h_descriptors_object, h_descriptors_scene; // cpu descriptors

    // feature point
    Ptr<cv::ORB> orb;
    if (desc_type == DESC_ORB) {
        const int MAX_FEATURES = 5000;
        orb = cv::ORB::create(MAX_FEATURES);
    }
    
    orb->detectAndCompute(h_object_image, noArray(), h_keypoints_object, h_descriptors_object);
    orb->detectAndCompute(h_scene_image, noArray(), h_keypoints_scene, h_descriptors_scene);

    // point match
    cv::DescriptorMatcher* matcher;
    if (m_type == M_BF) {
        matcher = new cv::BFMatcher(NORM_HAMMING);
    }

    if (m_type == M_FLANN) {
        matcher = new cv::FlannBasedMatcher(new flann::KDTreeIndexParams(6), new flann::SearchParams(10));
        h_descriptors_object.convertTo(h_descriptors_object, CV_32F);
        h_descriptors_scene.convertTo(h_descriptors_scene, CV_32F);
    }

    vector<vector<DMatch>> h_matches;
    matcher->knnMatch(h_descriptors_object, h_descriptors_scene, h_matches, 2);

    // filter match points
    std::vector<DMatch> h_good_matches;
    for (auto match : h_matches)
    {
        if ((match[0].distance < 0.89 * match[1].distance) && ((int)match.size() <= 2) && ((int)match.size() > 0))
        {
            h_good_matches.push_back(match[0]);
        }
    }

    // prepare data for homography
    vector<Point2f> h_pts_object, h_pts_scene;
    for (auto m : h_good_matches)
    {
        h_pts_object.push_back(h_keypoints_object.at(m.queryIdx).pt);
        h_pts_scene.push_back(h_keypoints_scene.at(m.trainIdx).pt);
    }

    // Homography Mat
    Mat h_M = cv::findHomography(h_pts_object, h_pts_scene, RHO, 0.5);
    h_mask = Mat::ones(h_object_image.size(), CV_8UC1);

    cv::warpPerspective(h_object_image, h_object_image, h_M, h_object_image.size(), INTER_LINEAR);
    cv::warpPerspective(h_mask, h_mask, h_M, h_mask.size(), INTER_LINEAR);
}


void process(Mat &x, Mat &mask)
{
    cv::medianBlur(x, x, 5);
    cv::normalize(x, x, 0, 128, NORM_MINMAX, -1, mask);
}

bool myfunc(Rect2i r1, Rect2i r2)
{
    return r1.area() > r2.area();
}

template<typename T>
std::vector<T> slice(std::vector<T> const &v, int m, int n)
{
    auto first = v.cbegin() + m;
    auto last = v.cbegin() + n + 1;
 
    std::vector<T> vec(first, last);
    return vec;
}
 

void detect(Mat &h_object_image, Mat &h_scene_image, Mat &h_mask)
{
    assert(h_object_image.channels() == 1);
    assert(h_scene_image.channels() == 1);
    
    cv::imwrite("before_obj.jpg", h_object_image);
    cv::imwrite("before_sce.jpg", h_scene_image);

    process(h_object_image, h_mask);
    process(h_scene_image, h_mask);

    cv::imwrite("process_obj.jpg", h_object_image);
    cv::imwrite("process_sce.jpg", h_scene_image);

    Mat h_diff_image;


    cv::subtract(h_object_image, h_scene_image, h_diff_image, h_mask);
    cv::abs(h_diff_image);
    CV:imwrite("diff.jpg", h_diff_image);

    Mat h_change_map;
    cv::threshold(h_diff_image, h_change_map, 10, 255, THRESH_BINARY_INV);
    
    
    cv::medianBlur(h_change_map, h_change_map, 7);

    Mat h_kernel = Mat::ones(3, 3, CV_8UC1);

    cv::erode(h_change_map, h_change_map, h_kernel);

    Mat labels, stats, centroids;
    cv::connectedComponentsWithStats(h_change_map, labels, stats, centroids);

    auto image_size = h_change_map.size();
    // cout << "stats.size()=" << stats.size() << endl;
    //std::cout << centroids << std::endl;

    vector<Rect2i> regions;
    Scalar color(255, 0, 0);

    Mat h_change_map_chaos;
    h_change_map.copyTo(h_change_map_chaos);
    for (int i = 1; i < stats.rows; i++)
    {
        int x = stats.at<int>(Point(0, i));
        int y = stats.at<int>(Point(1, i));
        int w = stats.at<int>(Point(2, i));
        int h = stats.at<int>(Point(3, i));
        int area = stats.at<int>(Point(4, i));
        Rect2i rect(x, y, w, h);
        cv::rectangle(h_change_map_chaos, rect, color);
        if (area < image_size.area() * 0.001 || area > image_size.area() * 0.1) continue;
        regions.push_back(rect);
    }
    cv::imwrite("chaos01.jpg", h_change_map_chaos);
    
    std::sort(regions.begin(), regions.end(), myfunc);

    int length = regions.size();
    // cout << length << endl;
    int reserve_length = min(length, max(2, length/2));  // take half of the regions no less than 2 (if regions size is less than 2, just take all)
    // cout << reserve_length << endl;

    vector<bool> checked(reserve_length, false);

    for (int i = 0; i < reserve_length - 1; i++)
    {
        if (checked[i])
            continue;
        for (int j = i + 1; j < reserve_length; j++)
        {
            if (checked[j])
                continue;

            if ((regions[i] & regions[j]).area() > 0)
            {
                regions[i] = regions[i] | regions[j];
                checked[j] = true;
            }
        }
        regions[i] *= 1.2;
    }

    
    for (int i = 0; i < reserve_length; i++)
    {
        if (!checked[i])
            cv::rectangle(h_change_map, regions[i], color);
    }

    cv::imwrite("change01.jpg", h_change_map);
}

int main()
{    
// input.txt example:
// sly_bjbmyw_16.jpg
// sly_bjbmyw_16_s1.jpg
// 1
// yw_gkxfw_18.jpg
// yw_gkxfw_18_s2.jpg
// 1

    // read data
    string line1, line2, line3;
    getline(cin, line1);
    getline(cin, line2);
    getline(cin, line3);

    Mat base_gray = imread(line1, IMREAD_GRAYSCALE); // base only use gray
    Mat target_color = imread(line2, IMREAD_COLOR); // needed for recognization step

    Mat target_gray, h_mask, h_object_gray, h_scene_gray;
    cv::cvtColor(target_color, target_gray, COLOR_BGR2GRAY); // target gray is used for registration and detection

    const int EPOCH = stoi(line3);
    auto start = chrono::steady_clock::now();
    for (int i = 0; i < EPOCH; i++)
    {
        base_gray.copyTo(h_object_gray);
        target_gray.copyTo(h_scene_gray);
        image_registration(h_object_gray, h_scene_gray, h_mask, DESC_ORB, M_FLANN);
        detect(h_object_gray, h_scene_gray, h_mask);
    }
    auto end = chrono::steady_clock::now();
    chrono::duration<double, milli> elapsed = end - start;
    cout << "time: " << elapsed.count() << "ms" << endl;

    cv::imwrite("gray.jpg", h_object_gray);
    // cv::split(h_im_out, chans);
    // cv::split(h_scene_color, scene_chans);
    // chans[2] = scene_chans[2];
    // cv::merge(chans, h_im_out);

    // imwrite("warp.jpg", h_im_out);
}