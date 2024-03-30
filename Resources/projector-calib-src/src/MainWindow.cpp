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


#include "MainWindow.hpp"

#include <QMessageBox>
#include <QFileDialog>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <QPainter>
#include <QIntValidator>
#include <QDoubleValidator>

#include <iostream>

#include "structured_light.hpp"

#include "Application.hpp"
#include "io_util.hpp"

#include "AboutDialog.hpp"


MainWindow::MainWindow(QWidget * parent, Qt::WindowFlags flags): 
    QMainWindow(parent, flags)
{
    setupUi(this);

    QAbstractItemModel & model = APP->model;
    image_tree->setModel(&model);
    connect(APP, SIGNAL(root_dir_changed(const QString&)), this, SLOT(_on_root_dir_changed(const QString&)));
    connect(image_tree->selectionModel(), SIGNAL(currentChanged(const QModelIndex &, const QModelIndex &)), 
                this, SLOT(_on_image_tree_currentChanged(const QModelIndex &, const QModelIndex &)));

    QSettings & config = APP->config;

    setWindowTitle(WINDOW_TITLE);

    //Decode group: 
    code_type_gray_radio->blockSignals(true);
    code_type_gray_radio->setChecked(true); //default and fixed!!
    code_type_gray_radio->blockSignals(false);
    shadow_threshold_spin->blockSignals(true);
    shadow_threshold_spin->setRange(0, 255);
    shadow_threshold_spin->setValue(config.value("main/shadow_threshold", DEFAULT_SHADOW_THRESHOLD).toUInt());
    shadow_threshold_spin->blockSignals(false);
    
    decode_group->setEnabled(false);
    decode_group->setVisible(false);
    //END Decode group


    b_line->blockSignals(true);
    b_line->setValidator(new QDoubleValidator(0.0, 1.0, 6, this));
    b_line->setText(config.value("robust_estimation/b", DEFAULT_ROBUST_B).toString());
    b_line->blockSignals(false);

    m_spin->blockSignals(true);
    m_spin->setRange(0, 255);
    m_spin->blockSignals(false);
    m_spin->setValue(config.value("robust_estimation/m", DEFAULT_ROBUST_M).toUInt());

    corner_count_x_spin->blockSignals(true);
    corner_count_x_spin->setRange(1, 255);
    corner_count_x_spin->blockSignals(false);
    corner_count_x_spin->setValue(config.value("main/corner_count_x", DEFAULT_CORNER_X).toUInt());

    corner_count_y_spin->blockSignals(true);
    corner_count_y_spin->setRange(1, 255);
    corner_count_y_spin->blockSignals(false);
    corner_count_y_spin->setValue(config.value("main/corner_count_y", DEFAULT_CORNER_Y).toUInt());

    corners_width_line->blockSignals(true);
    corners_width_line->setValidator(new QDoubleValidator(this));
    corners_width_line->setText(config.value("main/corners_width", DEFAULT_CORNER_WIDTH).toString());
    corners_width_line->blockSignals(false);

    corners_height_line->blockSignals(true);
    corners_height_line->setValidator(new QDoubleValidator(this));
    corners_height_line->setText(config.value("main/corners_height", DEFAULT_CORNER_HEIGHT).toString());
    corners_height_line->blockSignals(false);

    display_original_radio->blockSignals(true);
    display_original_radio->setChecked(true);
    display_original_radio->blockSignals(false);

    image2_label->setVisible(false);

    //set contextual menu for layout changing
    QAction * horizontal_action = new QAction("Horizontal", current_image_group);
    QAction * vertical_action = new QAction("Vertical", current_image_group);
    QActionGroup * layout_action_group = new QActionGroup(current_image_group);
    layout_action_group->addAction(horizontal_action);
    layout_action_group->addAction(vertical_action);
    horizontal_action->setCheckable(true);
    vertical_action->setCheckable(true);
    horizontal_action->setChecked(true);
    vertical_action->setChecked(false);
    connect(horizontal_action, SIGNAL(triggered(bool)), this, SLOT(_on_horizontal_layout_action_triggered(bool)));
    connect(vertical_action, SIGNAL(triggered(bool)), this, SLOT(_on_vertical_layout_action_triggered(bool)));
    current_image_group->addActions(layout_action_group->actions());
    current_image_group->setContextMenuPolicy(Qt::ActionsContextMenu);

    show_message("Ready");
}

MainWindow::~MainWindow()
{
}

void MainWindow::on_change_dir_action_triggered(bool checked)
{
    Application * app = APP;
    QString dirname = QFileDialog::getExistingDirectory(this, "Select Image Directory",
                           app->config.value("main/root_dir", QString()).toString(), QFileDialog::ReadOnly);

    if (dirname.isEmpty())
    {   //nothing selected
        return;
    }

    app->config.setValue("main/root_dir", dirname);
    app->set_root_dir(dirname);
}

void MainWindow::_on_image_tree_currentChanged(const QModelIndex & current, const QModelIndex & previous)
{
    //delete current image
    image1_label->clear();
    image2_label->clear();

    if (!current.isValid())
    {
        return;
    }

    Application * app = APP;

    unsigned level = 0, row = 0;
    QModelIndex parent = app->model.parent(current);

    if (parent.parent().isValid())
    {   //child
        level = parent.row();
        row   = current.row();
    }
    else
    {   //top level, select first child
        level = current.row();
        row   = 0;
    }
   
    cv::Mat image1, image2;

    //busy cursor
    QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
    QApplication::processEvents();
    
    if (display_original_radio->isChecked())
    {
        image1 = app->get_image(level, row, ColorImageRole);
    }

    if (display_decoded_radio->isChecked() && app->pattern_list.size()>level)
    {
        cv::Mat & pattern_image = app->pattern_list.at(level);
        image1 = sl::colorize_pattern(pattern_image, 0, 0x3ff);
        image2 = sl::colorize_pattern(pattern_image, 1, 0x2ff);
    }

    if (level<APP->chessboard_corners.size())
    {   //draw chessboard
        cv::Mat corners(APP->chessboard_corners.at(level));
        if (corners.rows>0)
        {
            if (image1.rows)
            {
                image1 = image1.clone();
                cv::drawChessboardCorners(image1, APP->chessboard_size, corners, true);
            }
            if (image2.rows)
            {
                image2 = image2.clone();
                cv::drawChessboardCorners(image2, APP->chessboard_size, corners, true);
            }
        }
    }

    //update the viewer
    image1_label->setPixmap(QPixmap::fromImage(io_util::qImage(image1)));
    image2_label->setPixmap(QPixmap::fromImage(io_util::qImage(image2)));

    //restore regular cursor
    QApplication::restoreOverrideCursor();
    QApplication::processEvents();
}

void MainWindow::show_message(const QString & message)
{
    if (!message.isEmpty())
    {
        statusBar()->showMessage(message);
    }
    else
    {
        statusBar()->clearMessage();
    }
    QApplication::processEvents();
}

void MainWindow::on_code_type_gray_radio_clicked(bool checked)
{
    APP->config.setValue("main/pattern_type", "gray");
}

void  MainWindow::on_shadow_threshold_spin_valueChanged(int i)
{
    APP->config.setValue("main/shadow_threshold", i);
}

void MainWindow::on_b_line_editingFinished()
{
    APP->config.setValue("robust_estimation/b", b_line->text().toDouble());
}

void MainWindow::on_m_spin_valueChanged(int i)
{
    APP->config.setValue("robust_estimation/m", i);
}

void MainWindow::on_corner_count_x_spin_valueChanged(int i)
{
    APP->config.setValue("main/corner_count_x", i);
}

void MainWindow::on_corner_count_y_spin_valueChanged(int i)
{
    APP->config.setValue("main/corner_count_y", i);
}

void MainWindow::on_corners_width_line_editingFinished()
{
    APP->config.setValue("main/corners_width", corners_width_line->text().toDouble());
}

void MainWindow::on_corners_height_line_editingFinished()
{
    APP->config.setValue("main/corners_height", corners_height_line->text().toDouble());
}

void MainWindow::on_quit_action_triggered(bool checked)
{
    close();
    APP->quit();
}

void MainWindow::on_save_vertical_image_action_triggered(bool checked)
{
    if (image1_label->pixmap()->isNull())
    {
        QMessageBox::critical(this, "Error", "Vertical image is empy.");
        return;
    }
    QString filename = QFileDialog::getSaveFileName(this, "Save vertical image", "saved_image_vertical.png", "Images (*.png)");
    if (!filename.isEmpty())
    {
        image1_label->pixmap()->save(filename);
        show_message(QString("Vertical Image saved: %1").arg(filename));
    }
}

void MainWindow::on_save_horizontal_image_action_triggered(bool checked)
{
    if (image2_label->pixmap()->isNull())
    {
        QMessageBox::critical(this, "Error", "Horizontal image is empy.");
        return;
    }
    QString filename = QFileDialog::getSaveFileName(this, "Save horizontal image", "saved_image_horizontal.png", "Images (*.png)");
    if (!filename.isEmpty())
    {
        image2_label->pixmap()->save(filename);
        show_message(QString("Horizontal Image saved: %1").arg(filename));
    }
}

void MainWindow::_on_root_dir_changed(const QString & dirname)
{
    //update user interface
    image1_label->clear();
    image2_label->clear();

    display_original_radio->blockSignals(true);
    display_original_radio->setChecked(true);
    display_original_radio->blockSignals(false);

    image2_label->setVisible(false);

    setWindowTitle(QString("%1 - %2").arg(WINDOW_TITLE, dirname));

    QModelIndex index = APP->model.index(0, 0);
    image_tree->selectionModel()->clearSelection();
    if (APP->model.rowCount()>0)
    {
        image_tree->blockSignals(true);
        image_tree->selectionModel()->select(index, QItemSelectionModel::SelectCurrent);
        _on_image_tree_currentChanged(index, index);
        image_tree->blockSignals(false);
        show_message(QString("%1 set read").arg(APP->model.rowCount()));
    }
}

void MainWindow::on_display_original_radio_clicked(bool checked)
{
    if (checked)
    {
        image2_label->setVisible(false);
        update_current_image();
    }
}

void MainWindow::on_display_decoded_radio_clicked(bool checked)
{
    if (checked)
    {
        image2_label->setVisible(true);
        update_current_image();
    }
}

void MainWindow::on_about_action_triggered(bool checked)
{
    AboutDialog dialog(this, Qt::WindowCloseButtonHint);
    dialog.exec();
}

void MainWindow::update_current_image(QModelIndex current)
{
    QModelIndex index = current;
    if (!index.isValid())
    {
        index = image_tree->selectionModel()->currentIndex();
    }
    if (!index.isValid())
    {
        index = APP->model.index(0, 0);
    }
    _on_image_tree_currentChanged(index, index);
}

void MainWindow::on_extract_corners_button_clicked(bool checked)
{
    show_message("Searching chessboard corners...");

    APP->cal_reset();
    APP->calibrationDialog.setWindowTitle("Corner detection");
    APP->calibrationDialog.show();
    QApplication::processEvents();

    APP->extract_chessboard_corners();
    update_current_image();

    APP->calibrationDialog.finish();
    APP->calibrationDialog.exec();
    APP->calibrationDialog.hide();
    QApplication::processEvents();

    show_message("Ready");
}

void MainWindow::on_calibrate_button_clicked(bool checked)
{
    show_message("Running calibration...");

    APP->cal_reset();
    APP->calibrationDialog.setWindowTitle("Calibration");
    APP->calibrationDialog.show();
    QApplication::processEvents();

    APP->calibrate();
    update_current_image();

    APP->calibrationDialog.finish();
    APP->calibrationDialog.exec();
    APP->calibrationDialog.hide();
    QApplication::processEvents();

    show_message("Ready");
}

void MainWindow::on_decode_button_clicked(bool checked)
{
    show_message("Decoding...");

    APP->cal_reset();
    APP->calibrationDialog.setWindowTitle("Decode");
    APP->calibrationDialog.show();
    QApplication::processEvents();

    APP->decode();
    update_current_image();

    APP->calibrationDialog.finish();
    APP->calibrationDialog.exec();
    APP->calibrationDialog.hide();
    QApplication::processEvents();

    show_message("Ready");
}

void MainWindow::_on_horizontal_layout_action_triggered(bool checked)
{
    QLayout *old_layout = current_image_group->layout();    
    if (old_layout)
    {
        delete old_layout;
    }
    QHBoxLayout *new_layout = new QHBoxLayout(current_image_group);
    new_layout->addWidget(image1_label);
    new_layout->addWidget(image2_label);
    current_image_group->setLayout(new_layout);
}

void MainWindow::_on_vertical_layout_action_triggered(bool checked)
{
    QLayout *old_layout = current_image_group->layout();
    if (old_layout)
    {
        delete old_layout;
    }
    QVBoxLayout *new_layout = new QVBoxLayout(current_image_group);
    new_layout->addWidget(image1_label);
    new_layout->addWidget(image2_label);
    current_image_group->setLayout(new_layout);
}
