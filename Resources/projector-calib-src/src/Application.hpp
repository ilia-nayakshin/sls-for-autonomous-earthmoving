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

#ifndef __APPLICATION_HPP__
#define __APPLICATION_HPP__

#ifndef WINVER
#define WINVER 0x0500
#define _WIN32_WINNT 0x0500
#endif

#include <QApplication>
#include <QSettings>
#include <QList>
#include <QFileSystemModel>
#include <QMap>

#include <opencv2/core/core.hpp>

#include "TreeModel.hpp"
#include "MainWindow.hpp"
#include "CalibrationDialog.hpp"

#if defined(_MSC_VER) && !defined(isnan)
#define isnan _isnan
#endif

enum Role {ImageFilenameRole = Qt::UserRole, GrayImageRole, ColorImageRole};

#define WINDOW_TITLE "Camera-Projector Calibration"
#define INI_FILENAME "projector-calib.ini"

//decode
#define DEFAULT_SHADOW_THRESHOLD    10
#define DEFAULT_PATTERN_TYPE        "gray"

//decode
#define DEFAULT_ROBUST_B    0.5
#define DEFAULT_ROBUST_M    5

//checkerboard size
#define DEFAULT_CORNER_X        7
#define DEFAULT_CORNER_Y        11
#define DEFAULT_CORNER_WIDTH    21.08
#define DEFAULT_CORNER_HEIGHT   21.00

class Application : public QApplication
{
    Q_OBJECT
public:
    Application(int & argc, char ** argv);
    ~Application();

    void set_root_dir(const QString & dirname);

    void clear(void);

    const cv::Mat get_image(unsigned level, unsigned n, Role role = GrayImageRole) const;

    bool extract_chessboard_corners(void);
    void decode(void);
    void calibrate(void);

    bool decode_gray_set(unsigned level, cv::Mat & pattern_image, cv::Mat & min_max_image) const;

    void load_config(void);

    inline void cal_set_current_message(const QString & text) const {calibrationDialog.set_current_message(text); processEvents();}
    inline void cal_reset(void) {calibrationDialog.reset(); processEvents();}
    inline void cal_set_progress_total(unsigned value) {calibrationDialog.set_progress_total(value); processEvents();}
    inline void cal_set_progress_value(unsigned value) {calibrationDialog.set_progress_value(value); processEvents();}
    inline void cal_message(const QString & text) const {calibrationDialog.message(text); processEvents();}
    inline bool cal_canceled(void) const {return calibrationDialog.canceled();}

public slots:
    void deinit(void);

Q_SIGNALS:
    void root_dir_changed(const QString & dirname);

public:
    QSettings  config;
    TreeModel  model;
    MainWindow mainWin;
    mutable CalibrationDialog calibrationDialog;

    cv::Mat cam_K;
    cv::Mat cam_kc;
    cv::Mat proj_K;
    cv::Mat proj_kc;
    cv::Mat R;
    cv::Mat T;

    cv::Size chessboard_size;
    cv::Size2f corner_size;
    std::vector<std::vector<cv::Point2f> > chessboard_corners;
    std::vector<std::vector<cv::Point2f> > projector_corners;
    std::vector<cv::Mat> pattern_list;
};

#define APP dynamic_cast<Application *>(Application::instance())

#endif  /* __APPLICATION_HPP__ */
