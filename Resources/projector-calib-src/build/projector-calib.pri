#
# vim:filetype=qmake sw=4 ts=4 expandtab
#

HEADERS += \
        $$SOURCEDIR/io_util.hpp \
        $$SOURCEDIR/Application.hpp \
        $$SOURCEDIR/MainWindow.hpp \
        $$SOURCEDIR/AboutDialog.hpp \
        $$SOURCEDIR/CalibrationDialog.hpp \
        $$SOURCEDIR/ImageLabel.hpp \
        $$SOURCEDIR/TreeModel.hpp \
        $$SOURCEDIR/structured_light.hpp \
        $$(NULL)

SOURCES += \
        $$SOURCEDIR/main.cpp \
        $$SOURCEDIR/io_util.cpp \
        $$SOURCEDIR/Application.cpp \
        $$SOURCEDIR/MainWindow.cpp \
        $$SOURCEDIR/AboutDialog.cpp \
        $$SOURCEDIR/CalibrationDialog.cpp \
        $$SOURCEDIR/ImageLabel.cpp \
        $$SOURCEDIR/TreeModel.cpp \
        $$SOURCEDIR/structured_light.cpp \
        $$(NULL)

FORMS = \
        $$FORMSDIR/MainWindow.ui \
        $$FORMSDIR/AboutDialog.ui \
        $$FORMSDIR/CalibrationDialog.ui \
        $$(NULL)
