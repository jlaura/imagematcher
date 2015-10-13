# -*- coding: utf-8 -*-
import argparse
import logging
import os

import numpy as np
import cv2

from log import log

def anorm2(a):
    return (a*a).sum(-1)
def anorm(a):
    return np.sqrt( anorm2(a) )
def getsize(img):
    h, w = img.shape[:2]
    return w, h

def parseargs():
    """
    Parse the commandline arguments
    """
    descriptors = ['sift', 'surf', 'orb']
    matchers = ['flann', 'brute']

    parser = argparse.ArgumentParser(description='ImageMatching Experimentation')
    parser.add_argument('-f', '--feature',
                        choices=descriptors,
                        default='sift', 
                        dest='descriptor',
                        help='The descripter to use' )
    parser.add_argument('-m', '--matcher',
                        choices=matchers,
                        default='flann',
                        dest='matcher',
                        help='The matcher to use')
    parser.add_argument('-p', '--matcher_parameters',
                        dest='matcher_params',
                        help='A parameter string or file (JSON or YAML) parameterization for the matcher')
    parser.add_argument('-d', '--descriptor_parameters',
                        dest='descriptor_params',
                        help='A parameter string or file (JSON or YAML) parameterization for the descriptor')
    parser.add_argument('--images',
                        nargs='*',
                        dest='images',
                        help='Comma separated list of images')
    parser.add_argument('--imagelist',
                        dest='imagelist',
                        help='A file containing a list of images')
    parser.add_argument('--minmatch',
                        dest='min_match_count',
                        default=10,
                        help='The minimum number of matches after outlier detection')

    return parser.parse_args()

#TODO: This should be in a matching module.
def init_feature(method, matcher, method_params={}, matcher_params={}):
    """
    Initialize an OpenCV feature matcher.
    
    Parameters
    ----------
    method : {'sift', 'surf', 'orb', 'akaze', 'brisk'}
             Then method to be used
    
    matcher : {'flann', 'brute'}
              The matcher to be used

    method_params : dict
                    A dictionary of parameters for the given method

    matcher_params : dict
                     A dictionary of parameters for the given matcher

    Returns
    -------
    detector : object
               An OpenCV feature detector

    matcher : object
              An OpenCV matcher
    """

    FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
    FLANN_INDEX_LSH    = 6
    print method_params

    if method == 'sift':
        detector = cv2.xfeatures2d.SIFT_create(**method_params)
        norm = cv2.NORM_L2 #Euclidean Distance
    elif method == 'surf':
        detector = cv2.xfeatures2d.SURF_create(800)
        norm = cv2.NORM_L2
    elif method == 'orb':
        detector = cv2.ORB_create(400)
        norm = cv2.NORM_HAMMING
    elif method == 'akaze':
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    elif method == 'brisk':
        detector = cv2.BRISK_create()
        norm = cv2.NORM_HAMMING
    lse:
        return None, None
    if matcher == 'flann':
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    elif matcher == 'brute':
        matcher = cv2.BFMatcher(norm)
    return detector, matcher

def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)

    cv2.imshow(win, vis)

    def onmouse(event, x, y, flags, param):
        cur_vis = vis
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cur_vis = vis0.copy()
            r = 8
            m = (anorm(p1 - (x, y)) < r) | (anorm(p2 - (x, y)) < r)
            idxs = np.where(m)[0]
            kp1s, kp2s = [], []
            for i in idxs:
                 (x1, y1), (x2, y2) = p1[i], p2[i]
                 col = (red, green)[status[i]]
                 cv2.line(cur_vis, (x1, y1), (x2, y2), col)
                 kp1, kp2 = kp_pairs[i]
                 kp1s.append(kp1)
                 kp2s.append(kp2)
            cur_vis = cv2.drawKeypoints(cur_vis, kp1s, 4, color=kp_color)
            cur_vis[:,w1:] = cv2.drawKeypoints(cur_vis[:,w1:], kp2s, 4, color=kp_color)

        cv2.imshow(win, cur_vis)
    cv2.setMouseCallback(win, onmouse)
    return vis

def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs

def match(matcher, kp1, desc1, kp2, desc2, ratio=0.7):
    """
    Parameters
    ----------
    
    matcher : obj
              An OpenCV Matcher object

    desc1 : array
            nfeatures x ???

    desc2 : array
            nfeatures x ???

    Returns
    -------
    p1 : array
         of ???
    
    p2 : array
         of ???

    kp_pairs : list
               of Keypoint pair tuples, these are all pts pre-outlier removal

    raw_matches : list
                  of DMatch pair tuples
    """
    raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)
    p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches, ratio=ratio) #Lowe ratio
   
    return p1, p2, kp_pairs, raw_matches

#TODO: Move to a IO module
def parse_yaml(yaml_file):
    """
    Parse a YAML configuration file into a dict

    Parameters
    ----------
    yaml_file : obj
                File handle to the file to be parsed
    Returns
    -------
    config : dict
             Parsed configuration file
    """

    import yaml #Could be a conditional dependency.
    config = yaml.load(yaml_file.read())

    return config

#TODO: Move to IO module
def parse_json(json_file):
    """
    Parse a JSON configuration file

    Parameters
    ----------
    json_file : obj
                File handle to the file to be parsed

    Returns
    -------
    config : dict
             Parsed configuration file
    """
    config = json.load(json_file)

    return config

def parse_configuration_file(path):
    if os.path.exists(path):
        logtype = os.path.splitext(os.path.basename(path))[1]
        with open(path, 'rt') as f:
            if logtype == '.json':
                return parse_json(f)
            else:
                return parse_yaml(f)

def main(args):
    logger = logging.getLogger(__name__)
    #TODO: Code to accept string, JSON, or YAML parameterization

    if args.matcher_params:
        matcher_params = parse_configuration_file(args.matcher_params)
    else:
        matcher_params = {}
    if args.descriptor_params:
        descriptor_params = parse_configuration_file(args.descriptor_params)
    else:
        descriptor_params = {}
    fn1, fn2 = args.images 
    
    logger.info('Processing: {}'.format(fn1))
    logger.info('Processing: {}'.format(fn2))
     
    img1 = cv2.imread(fn1, 0)
    img2 = cv2.imread(fn2, 0)

    detector, matcher = init_feature(args.descriptor, args.matcher, matcher_params, descriptor_params)

    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)
    
    logger.info('img1: {} features img2: {} features'.format(len(kp1), len(kp2)))

    #Match the features
    p1, p2, kp_pairs, raw_matches = match(matcher, kp1, desc1, kp2, desc2)
     
    #If enough matches were found, run RANSAC
    #TODO: No reason to hard code
    if len(p1) > 10:
        transformation, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        logger.info('{} / {} inliers / matched'.format(np.sum(status), len(status))) 
        transformation, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        logger.info('{} / {} inliers / matched'.format(np.sum(status), len(status)))
        explore_match('match', img1, img2, kp_pairs, status=status, H=transformation)
        cv2.waitKey()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parseargs()
    log.setup_logging('logging.json')
    main(args)
