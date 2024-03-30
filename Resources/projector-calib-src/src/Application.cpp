/*
Copyright (c) 2012, Daniel Moreno and Gabriel Taubin
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Brown University nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL DANIEL MORENO AND GABRIEL TAUBIN BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#define _USE_MATH_DEFINES
#include <cmath> 

#include "Application.hpp"

#include <QDir>
#include <QProgressDialog>
#include <QMessageBox>

#include <iostream>
#include <ctime>
#include <cstdlib>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "structured_light.hpp"
//#include "io_util.hpp"

static const float DEFAULT_B = 0.3f;
static const float DEFAULT_M = 5;

Application::Application(int & argc, char ** argv) : 
    QApplication(argc, argv),
    config(/*QDesktopServices::storageLocation(QDesktopServices::HomeLocation) + "/"*/INI_FILENAME, QSettings::IniFormat, this),
    model(this),
    mainWin((QWidget*)(load_config(), NULL)),
    calibrationDialog(&mainWin, Qt::Window|Qt::CustomizeWindowHint|Qt::WindowTitleHint),
    cam_K(),
    cam_kc(),
    proj_K(),
    proj_kc(),
    R(),
    T(),
    chessboard_size(11, 7),
    corner_size(21.f, 21.f),
    chessboard_corners(),
    projector_corners(),
    pattern_list()
{
    connect(this, SIGNAL(aboutToQuit()), this, SLOT(deinit()));

    //setup the main window state
    mainWin.show();
    mainWin.restoreGeometry(config.value("main/window_geometry").toByteArray());
    QVariant window_state = config.value("main/window_state");
    if (window_state.isValid())
    {
        mainWin.setWindowState(static_cast<Qt::WindowStates>(window_state.toUInt()));
    }

    //set model
    set_root_dir(config.value("main/root_dir", QDir::currentPath()).toString());
    QModelIndex index = model.index(0, 0);
    mainWin._on_image_tree_currentChanged(index, index);
}

Application::~Application()
{
}

void Application::deinit(void)
{
    config.setValue("main/window_geometry", mainWin.saveGeometry());
    config.setValue("main/window_state", static_cast<unsigned>(mainWin.windowState()));
}

void Application::clear(void)
{
    cam_K = cv::Mat();
    cam_kc = cv::Mat();
    proj_K = cv::Mat();
    proj_kc = cv::Mat();
    R = cv::Mat();
    T = cv::Mat();

    chessboard_corners.clear();
    projector_corners.clear();
    pattern_list.clear();
}

void Application::load_config(void)
{
    //decode
    if (!config.value("main/pattern_type").isValid())
    {
        config.setValue("main/pattern_type", DEFAULT_PATTERN_TYPE);
    }
    if (!config.value("main/shadow_threshold").isValid())
    {
        config.setValue("main/shadow_threshold", DEFAULT_SHADOW_THRESHOLD);
    }

    //robust estimation
    if (!config.value("robust_estimation/b").isValid())
    {
        config.setValue("robust_estimation/b", DEFAULT_ROBUST_B);
    }
    if (!config.value("robust_estimation/m").isValid())
    {
        config.setValue("robust_estimation/m", DEFAULT_ROBUST_M);
    }

    //checkerboard size
    if (!config.value("main/corner_count_x").isValid())
    {
        config.setValue("main/corner_count_x", DEFAULT_CORNER_X);
    }
    if (!config.value("main/corner_count_y").isValid())
    {
        config.setValue("main/corner_count_y", DEFAULT_CORNER_Y);
    }
    if (!config.value("main/corners_width").isValid())
    {
        config.setValue("main/corners_width", DEFAULT_CORNER_WIDTH);
    }
    if (!config.value("main/corners_height").isValid())
    {
        config.setValue("main/corners_height", DEFAULT_CORNER_HEIGHT);
    }
}

void Application::set_root_dir(const QString & dirname)
{
    QDir root_dir(dirname);

    //reset internal data
    model.clear();
    clear();

    QStringList dirlist = root_dir.entryList(QDir::Dirs|QDir::NoDotAndDotDot, QDir::Name);
    foreach (const QString & item, dirlist)
    {
        QDir dir(root_dir.filePath(item));

        QStringList filters;
        filters << "*.jpg" << "*.bmp" << "*.png";

        QStringList filelist = dir.entryList(filters, QDir::Files, QDir::Name);
        QString path = dir.path();

        //setup the model
        int filecount = filelist.count();

        if (filecount<1)
        {   //no images, skip
            continue;
        }

        unsigned row = model.rowCount();
        if (!model.insertRow(row))
        {
            std::cout << "Failed model insert " << item.toStdString() << "("<< row << ")" << std::endl;
            continue;
        }

        //add the childrens
        QModelIndex parent = model.index(row, 0);
        model.setData(parent, item,  Qt::DisplayRole);

        for (int i=0; i<filecount; i++)
        {
            const QString & filename = filelist.at(i);
            if (!model.insertRow(i, parent))
            {
                std::cout << "Failed model insert " << filename.toStdString() << "("<< row << ")" << std::endl;
                break;
            }

            QModelIndex index = model.index(i, 0, parent);
            model.setData(index, QString("#%1 %2").arg(i, 2, 10, QLatin1Char('0')).arg(filename), Qt::DisplayRole);

            //additional data
            model.setData(index, path + "/" + filename, ImageFilenameRole);
        }
    }

    emit root_dir_changed(dirname);
}

const cv::Mat Application::get_image(unsigned level, unsigned n, Role role) const
{
    if (role!=GrayImageRole && role!=ColorImageRole)
    {   //invalid args
        return cv::Mat();
    }

    //try to load
    if (model.rowCount()<static_cast<int>(level))
    {   //out of bounds
        return cv::Mat();
    }
    QModelIndex parent = model.index(level, 0);
    if (model.rowCount(parent)<static_cast<int>(n))
    {   //out of bounds
        return cv::Mat();
    }

    QModelIndex index = model.index(n, 0, parent);
    if (!index.isValid())
    {   //invalid index
        return cv::Mat();
    }

    QString filename = model.data(index, ImageFilenameRole).toString();
    std::cout << "[" << (role==GrayImageRole ? "gray" : "color") << "] Filename: " << filename.toStdString() << std::endl;

    //load image
    cv::Mat rgb_image = cv::imread(filename.toStdString());
    if (rgb_image.rows>0 && rgb_image.cols>0)
    {
        //color
        if (role==ColorImageRole)
        {
            return rgb_image;
        }
        
        //gray scale
        if (role==GrayImageRole)
        {
            cv::Mat gray_image;
            cvtColor(rgb_image, gray_image, CV_BGR2GRAY);
            return gray_image;
        }
    }

    return cv::Mat();
}

bool Application::extract_chessboard_corners(void)
{
    chessboard_size = cv::Size(config.value("main/corner_count_x").toUInt(), config.value("main/corner_count_y").toUInt()); //interior number of corners
    corner_size = cv::Size2f(config.value("main/corners_width").toDouble(), config.value("main/corners_height").toDouble());

    unsigned count = static_cast<unsigned>(model.rowCount());

    cal_set_progress_total(count);
    cal_set_progress_value(0);
    cal_set_current_message("Extracting corners...");

    chessboard_corners.clear();
    chessboard_corners.resize(count);

    cv::Size imageSize;
    int image_scale = 1;

    bool all_found = true;
    for (unsigned i=0; i<count; i++)
    {
        QString set_name = model.data(model.index(i, 0), Qt::DisplayRole).toString();
        cal_set_current_message(QString("Extracting corners... %1").arg(set_name));

        cv::Mat gray_image = get_image(i, 0, GrayImageRole);
        if (gray_image.rows<1)
        {
            continue;
        }

        if (i==0)
        {   //init image size
            imageSize = gray_image.size();
            if (imageSize.width>1024)
            {
                image_scale = imageSize.width/1024;
            }
        }
        else if (imageSize != gray_image.size())
        {   //error
            std::cout << "ERROR: image of different size: set " << i << std::endl;
            return false;
        }

        cv::Mat small_img;

        if (image_scale>1)
        {
            cv::resize(gray_image, small_img, cv::Size(gray_image.cols/image_scale, gray_image.rows/image_scale));
        }

        if (cal_canceled())
        {
            cal_set_current_message("Extract corners canceled");
            cal_message("Extract corners canceled");
            return false;
        }

        //this will be filled by the detected corners
        std::vector<cv::Point2f> & corners = chessboard_corners[i];
        if (cv::findChessboardCorners(small_img, chessboard_size, corners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE /*+ cv::CALIB_CB_FILTER_QUADS*/))
        {
            cal_message(QString(" * %1: found %2 corners").arg(set_name).arg(corners.size()));
            std::cout << " - corners: " << corners.size() << std::endl;
        }
        else
        {
            all_found = false;
            cal_message(QString(" * %1: chessboard not found!").arg(set_name));
            std::cout << " - chessboard not found!" << std::endl;
        }

        for (std::vector<cv::Point2f>::iterator iter=corners.begin(); iter!=corners.end(); iter++)
        {
            *iter = image_scale*(*iter);
        }
        if (corners.size())
        {
            cv::cornerSubPix(gray_image, corners, cv::Size(11, 11), cv::Size(-1, -1), 
                                cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
        }

        cal_set_progress_value(i+1);
    }

    cal_set_current_message("Extract corners finished");
    cal_set_progress_value(count);
    return all_found;
}

void Application::decode(void)
{
    unsigned count = static_cast<unsigned>(model.rowCount());
    cv::Size imageSize;

    cal_set_progress_total(count);
    cal_set_progress_value(0);
    cal_set_current_message("Decoding...");

    pattern_list.resize(count);

    QString path = config.value("main/root_dir").toString();
 
    //decode gray patterns
    for (unsigned i=0; i<count; i++)
    {
        QString set_name = model.data(model.index(i, 0), Qt::DisplayRole).toString();
        cal_set_current_message(QString("Decoding... %1").arg(set_name));

        cv::Mat & pattern_image = pattern_list[i];
        cv::Mat min_max_image;
        if (!decode_gray_set(i, pattern_image, min_max_image))
        {   //error
            std::cout << "ERROR: Decode image set " << i << " failed. " << std::endl;
            return;
        }

        if (cal_canceled())
        {
            cal_set_current_message("Decode canceled");
            cal_message("Decode canceled");
            return;
        }

        if (i==0)
        {
            imageSize = pattern_image.size();
        }
        else if (imageSize != pattern_image.size())
        {
            cal_message(QString("ERROR: pattern image of different size: set %1").arg(set_name));
            std::cout << "ERROR: pattern image of different size: set " << i << std::endl;
            return;
        }

        //save pattern image as PGM for debugging
        //QString filename = path + "/" + set_name;
        //io_util::write_pgm(pattern_image, qPrintable(filename));

        cal_message(QString(" * %1: decoded").arg(set_name));
        cal_set_progress_value(i+1);
    }

    cal_set_current_message("Decode finished");
    cal_set_progress_value(count);
}

void Application::calibrate(void)
{   //try to calibrate the camera, projector, and stereo system

    unsigned count = static_cast<unsigned>(model.rowCount());

    const unsigned threshold = config.value("main/shadow_threshold", 0).toUInt();

    std::cout << " shadow_threshold = " << threshold << std::endl;

    cv::Size imageSize;

    //detect corners ////////////////////////////////////
    cal_message("Extracting corners:");
    if (!extract_chessboard_corners())
    {
        return;
    }
    cal_message("");
    
    //generate world object coordinates
    std::vector<cv::Point3f> world_corners;
    for (int h=0; h<chessboard_size.height; h++)
    {
        for (int w=0; w<chessboard_size.width; w++)
        {
            world_corners.push_back(cv::Point3f(corner_size.width * w, corner_size.height * h, 0.f));
        }
    }
    
    std::vector<std::vector<cv::Point3f> > objectPoints;
    objectPoints.reserve(count);
    for (unsigned i=0; i<count; i++)
    {
        objectPoints.push_back(world_corners);
    }

    //collect projector correspondences
    projector_corners.resize(count);
    pattern_list.resize(count);

    cal_set_progress_total(count);
    cal_set_progress_value(0);
    cal_set_current_message("Decoding and computing homographies...");

    for (unsigned i=0; i<count; i++)
    {
        std::vector<cv::Point2f> const& corners = chessboard_corners[i];
        std::vector<cv::Point2f> & pcorners = projector_corners[i];
        pcorners.clear(); //erase previous points

        QString set_name = model.data(model.index(i, 0), Qt::DisplayRole).toString();
        cal_set_current_message(QString("Decoding... %1").arg(set_name));

        cv::Mat & pattern_image = pattern_list[i];
        cv::Mat min_max_image;
        if (!decode_gray_set(i, pattern_image, min_max_image))
        {   //error
            std::cout << "ERROR: Decode image set " << i << " failed. " << std::endl;
            return;
        }

        if (i==0)
        {
            imageSize = pattern_image.size();
        }
        else if (imageSize != pattern_image.size())
        {
            std::cout << "ERROR: pattern image of different size: set " << i << std::endl;
            return;
        }

        //cv::Mat out_pattern_image = sl::PIXEL_UNCERTAIN*cv::Mat::ones(pattern_image.size(), pattern_image.type());

        cal_set_current_message(QString("Computing homographies... %1").arg(set_name));

        for (std::vector<cv::Point2f>::const_iterator iter=corners.begin(); iter!=corners.end(); iter++)
        {
            const cv::Point2f & p = *iter;
            cv::Point2f q;

            if (cal_canceled())
            {
                cal_set_current_message("Calibration canceled");
                cal_message("Calibration canceled");
                return;
            }
            processEvents();

            //find an homography around p
            unsigned WINDOW_SIZE = 30;
            std::vector<cv::Point2f> img_points, proj_points;
            if (p.x>WINDOW_SIZE && p.y>WINDOW_SIZE && p.x+WINDOW_SIZE<pattern_image.cols && p.y+WINDOW_SIZE<pattern_image.rows)
            {
                for (unsigned h=p.y-WINDOW_SIZE; h<p.y+WINDOW_SIZE; h++)
                {
                    register const cv::Vec2f * row = pattern_image.ptr<cv::Vec2f>(h);
                    register const cv::Vec2b * min_max_row = min_max_image.ptr<cv::Vec2b>(h);
                    //cv::Vec2f * out_row = out_pattern_image.ptr<cv::Vec2f>(h);
                    for (unsigned w=p.x-WINDOW_SIZE; w<p.x+WINDOW_SIZE; w++)
                    {
                        const cv::Vec2f & pattern = row[w];
                        const cv::Vec2b & min_max = min_max_row[w];
                        //cv::Vec2f & out_pattern = out_row[w];
                        if (sl::INVALID(pattern))
                        {
                            continue;
                        }
                        if ((min_max[1]-min_max[0])<static_cast<int>(threshold))
                        {   //apply threshold and skip
                            continue;
                        }

                        img_points.push_back(cv::Point2f(w, h));
                        proj_points.push_back(cv::Point2f(pattern));

                        //out_pattern = pattern;
                    }
                }
                cv::Mat H = cv::findHomography(img_points, proj_points);
                //std::cout << " H:\n" << H << std::endl;
                cv::Point3d Q = cv::Point3d(cv::Mat(H*cv::Mat(cv::Point3d(p.x, p.y, 1.0))));
                q = cv::Point2f(Q.x/Q.z, Q.y/Q.z);
            }
            else
            {
                return;
            }

            //save
            pcorners.push_back(q);
        }

        cal_message(QString(" * %1: finished").arg(set_name));
        cal_set_progress_value(i+1);
    }
    cal_message("");

    int cal_flags = 0
                  //+ cv::CALIB_FIX_K1
                  //+ cv::CALIB_FIX_K2
                  //+ cv::CALIB_ZERO_TANGENT_DIST
                  + cv::CALIB_FIX_K3
                  ;
    
    //calibrate the camera ////////////////////////////////////
    cal_message(" * Calibrate camera");
    std::vector<cv::Mat> cam_rvecs, cam_tvecs;
    int cam_flags = cal_flags;
    double cam_error = cv::calibrateCamera(objectPoints, chessboard_corners, imageSize, cam_K, cam_kc, cam_rvecs, cam_tvecs, cam_flags, 
                                            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, DBL_EPSILON));

    //calibrate the projector ////////////////////////////////////
    cal_message(" * Calibrate projector");
    std::vector<cv::Mat> proj_rvecs, proj_tvecs;
    int proj_flags = cal_flags;
    double proj_error = cv::calibrateCamera(objectPoints, projector_corners, sl::projector_size, proj_K, proj_kc, proj_rvecs, proj_tvecs, proj_flags, 
                                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, DBL_EPSILON));

    //stereo calibration
    cal_message(" * Calibrate stereo");
    cv::Mat E, F;
    double stereo_error = cv::stereoCalibrate(objectPoints, chessboard_corners, projector_corners, cam_K, cam_kc, proj_K, proj_kc, imageSize /*ignored*/, R, T, E, F, 
                                                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, DBL_EPSILON), 
                                                cv::CALIB_FIX_INTRINSIC /*cv::CALIB_USE_INTRINSIC_GUESS + cal_flags*/);

    cal_message("\n **** Calibration results ****\n");
    std::stringstream stream;

    //print
    std::cout << "Camera Calib results: " << std::endl
        << " - reprojection error: " << cam_error << std::endl
        << " - K:\n" << cam_K << std::endl
        << " - kc: " << cam_kc << std::endl
        //<< " - Rvecs:\n" << cam_rvecs << std::endl
        //<< " - Tvecs:\n" << cam_tvecs << std::endl
        ;
    stream << "Camera: " << std::endl
        << " - reprojection error: " << cam_error << std::endl
        << " - K:\n" << cam_K << std::endl
        << " - kc: " << cam_kc << std::endl
        //<< " - Rvecs:\n" << cam_rvecs << std::endl
        //<< " - Tvecs:\n" << cam_tvecs << std::endl
        << std::endl
        ;

    std::cout << "Projector Calib results: " << std::endl
        << " - reprojection error: " << proj_error << std::endl
        << " - K:\n" << proj_K << std::endl
        << " - kc: " << proj_kc << std::endl
        //<< " - Rvecs:\n" << proj_rvecs << std::endl
        //<< " - Tvecs:\n" << proj_tvecs << std::endl
        ;
    stream << "Projector: " << std::endl
        << " - reprojection error: " << proj_error << std::endl
        << " - K:\n" << proj_K << std::endl
        << " - kc: " << proj_kc << std::endl
        //<< " - Rvecs:\n" << proj_rvecs << std::endl
        //<< " - Tvecs:\n" << proj_tvecs << std::endl
        << std::endl
        ;

    std::cout << "Stereo Calib results: " << std::endl
        << " - reprojection error: " << stereo_error << std::endl
        << " - R:\n" << R << std::endl
        << " - T:\n" << T << std::endl
        ;
    stream << "Stereo: " << std::endl
        << " - reprojection error: " << stereo_error << std::endl
        << " - R:\n" << R << std::endl
        << " - T: " << T << std::endl
        ;

    cal_message(QString(stream.str().c_str()));

    //save to file
    QString path = config.value("main/root_dir").toString();
    QString filename = path + "/calibration.yml";
    cv::FileStorage fs(filename.toStdString(), cv::FileStorage::WRITE);
    fs << "cam_K" << cam_K << "cam_kc" << cam_kc
       << "proj_K" << proj_K << "proj_kc" << proj_kc
       << "R" << R << "T" << T
       << "b" << config.value("robust_estimation/b", DEFAULT_B).toFloat() 
       << "m" << config.value("robust_estimation/m", DEFAULT_M).toInt()
       << "stereo_error" << stereo_error
       ;
    fs.release();
    cal_message(QString("Calibration saved: %1").arg(filename));

    //save corners
    FILE * fp = NULL;
    
    filename = path + "/model.txt";
    fp = fopen(qPrintable(filename), "w");
    if (!fp)
    {
        std::cout << "ERROR: could no open " << filename.toStdString() << std::endl;
        return;
    }
    std::cout << "Saved " << filename.toStdString() << std::endl;
    for (std::vector<cv::Point3f>::const_iterator iter=world_corners.begin(); iter!=world_corners.end(); iter++)
    {
        fprintf(fp, "%lf %lf %lf\n", iter->x, iter->y, iter->z);
    }
    fclose(fp);
    fp = NULL;

    for (unsigned i=0; i<count; i++)
    {
        std::vector<cv::Point2f> & corners = chessboard_corners[i];
        std::vector<cv::Point2f> & pcorners = projector_corners[i];

        QString filename1 = QString("%1/cam_%2.txt").arg(path).arg(i, 2, 10, QLatin1Char('0'));
        FILE * fp1 = fopen(qPrintable(filename1), "w");
        if (!fp1)
        {
            std::cout << "ERROR: could no open " << filename1.toStdString() << std::endl;
            return;
        }
        QString filename2 = QString("%1/proj_%2.txt").arg(path).arg(i, 2, 10, QLatin1Char('0'));
        FILE * fp2 = fopen(qPrintable(filename2), "w");
        if (!fp2)
        {
            fclose(fp1);
            std::cout << "ERROR: could no open " << filename2.toStdString() << std::endl;
            return;
        }

        std::cout << "Saved " << filename1.toStdString() << std::endl;
        std::cout << "Saved " << filename2.toStdString() << std::endl;

        std::vector<cv::Point2f>::const_iterator iter1 = corners.begin();
        std::vector<cv::Point2f>::const_iterator iter2 = pcorners.begin();
        for (unsigned j=0; j<corners.size(); j++, iter1++, iter2++)
        {
            fprintf(fp1, "%lf %lf\n", iter1->x, iter1->y);
            fprintf(fp2, "%lf %lf\n", iter2->x, iter2->y);
        }
        fclose(fp1);
        fclose(fp2);
    }

    cal_message("Calibration finished");
}

bool Application::decode_gray_set(unsigned level, cv::Mat & pattern_image, cv::Mat & min_max_image) const
{
    if (model.rowCount()<static_cast<int>(level))
    {   //out of bounds
        return false;
    }

    if (cal_canceled())
    {
        cal_set_current_message("Decode canceled");
        cal_message("Decode canceled");
        return false;
    }
    processEvents();

    //estimate direct component
    const float b = config.value("robust_estimation/b", DEFAULT_B).toFloat();
    std::vector<cv::Mat> images;
    QList<unsigned> direct_component_images(QList<unsigned>() << 15 << 16 << 17 << 18 << 35 << 36 << 37 << 38);
    foreach (unsigned i, direct_component_images)
    {
        images.push_back(get_image(level, i-1));
    }
    cv::Mat direct_light = sl::estimate_direct_light(images, b);
    cal_message("Estimate direct and global light components... done.");

    std::vector<std::string> image_names;

    QModelIndex parent = model.index(level, 0);
    unsigned level_count = static_cast<unsigned>(model.rowCount(parent));
    for (unsigned i=0; i<level_count; i++)
    {
        QModelIndex index = model.index(i, 0, parent);
        std::string filename = model.data(index, ImageFilenameRole).toString().toStdString();
        std::cout << "[decode_set " << level << "] Filename: " << filename << std::endl;

        image_names.push_back(filename);
    }

    if (cal_canceled())
    {
        cal_set_current_message("Decode canceled");
        cal_message("Decode canceled");
        return false;
    }
    processEvents();

    cal_message("Decoding, please wait...");
    const unsigned m = config.value("robust_estimation/m", DEFAULT_M).toUInt();
    return sl::decode_pattern(image_names, pattern_image, min_max_image, sl::RobustDecode|sl::GrayPatternDecode, direct_light, m);
}
