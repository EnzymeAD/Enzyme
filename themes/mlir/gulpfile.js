'use strict';

var gulp = require('gulp');
var $ = require('gulp-load-plugins')();

require('es6-promise').polyfill();

var webpack = require("webpack");
var webpackStream = require("webpack-stream");
var webpackConfig = require("./webpack.config");

var runSequence = require('run-sequence');

var browserSync = require('browser-sync').create();
var reload = browserSync.reload;

var src_paths = {
  sass: ['src/scss/*.scss'],
  script: ['src/js/*.js'],
};

var dest_paths = {
  style: 'static/css/',
  script: 'static/js/',
  browserSync: ''
};


gulp.task('lint:sass', function() {
  return gulp.src(src_paths.sass)
    .pipe($.plumber({
      errorHandler: function(err) {
        console.log(err.messageFormatted);
        this.emit('end');
      }
    }))
    .pipe($.stylelint({
      config: {
        ignoreFiles: "src/scss/_normalize.scss",
        extends: [
          "stylelint-config-recommended",
          "stylelint-scss",
          "stylelint-config-recommended-scss"
        ],
        rules: {
          "block-no-empty": null,
          "no-descending-specificity": null
        }
      },
      reporters: [
        {
          formatter: 'string',
          console: true
        }
      ]
    }));
});

gulp.task('sass:style', function() {
  return gulp.src(src_paths.sass)
    .pipe($.plumber({
      errorHandler: function(err) {
        console.log(err.messageFormatted);
        this.emit('end');
      }
    }))
    .pipe($.sass({
      outputStyle: 'expanded'
    }).on( 'error', $.sass.logError ))
    .pipe($.autoprefixer({
        browsers: ['last 2 versions'],
        cascade: false
    }))
    .pipe(gulp.dest(dest_paths.style))
    .pipe($.cssnano())
    .pipe($.rename({ suffix: '.min' }))
    .pipe(gulp.dest(dest_paths.style));
});

gulp.task('lint:javascript', function() {
  return gulp.src(dest_paths.script)
    .pipe($.jshint())
    .pipe($.jshint.reporter('jshint-stylish'));
});

gulp.task('lint:eslint', function() {
  return gulp.src(src_paths.script)
    .pipe($.eslint.format())
    .pipe($.eslint.failAfterError());
});

gulp.task('webpack', function() {
  return webpackStream(webpackConfig, webpack)
    .on('error', function (e) {
      this.emit('end');
    })
    .pipe(gulp.dest("dist"));
});

gulp.task('lint', ['lint:sass', 'lint:eslint', 'lint:javascript']);
gulp.task('sass', ['sass:style']);
gulp.task('script', ['webpack']);

gulp.task('default', function(callback) {
  runSequence(
    'lint',
    'sass',
    'script',
    callback
  );
});

gulp.task('watch', function() {
  gulp.watch([src_paths.sass, src_paths.script], ['default']);
});
