import easyaccess as ea
import threading
import os
import time
import json
import glob

STATUS_OK = 'ok'
STATUS_ERROR = 'error'
DEFAULT_RESPONSE = {
    'status': STATUS_OK,
    'msg': '',
    'elapsed': 0.0,
    'data': {},
    'files': [],
    'sizes': []
}

def get_filesize(filename):
    size = os.path.getsize(filename)
    size = size * 1. / 1024.
    if size > 1024. * 1024:
        size = '%.2f GB' % (1. * size / 1024. / 1024)
    elif size > 1024.:
        size = '%.2f MB' % (1. * size / 1024.)
    else:
        size = '%.2f KB' % (size)
    return size


def check_query(query, db, username, password):
    response = DEFAULT_RESPONSE
    try:
        connection = ea.connect(db, user=username, passwd=password)
        cursor = connection.cursor()
    except Exception as e:
        response['status'] = STATUS_ERROR
        response['msg'] = str(e).strip()
        return response
    try:
        cursor.parse(query.encode())
    except Exception as e:
        response['status'] = STATUS_ERROR
        response['msg'] = str(e).strip()
    cursor.close()
    connection.close()
    return response


def run_quick(query, db, username, password):
    response = DEFAULT_RESPONSE
    try:
        connection = ea.connect(db, user=username, passwd=password)
        cursor = connection.cursor()

        # Start the clock
        tt = threading.Timer(25, connection.con.cancel)
        tt.start()

        if query.lower().lstrip().startswith('select'):
            try:
                df = connection.query_to_pandas(query)
                # df.to_csv(os.path.join(user_folder, 'quickResults.csv'), index=False)
                df = df[0:1000]
                data = df.to_json(orient='records')
                response['data'] = data
            except Exception as e:
                err_out = str(e).strip()
                if 'ORA-01013' in err_out:
                    err_out = 'Time Exceeded (30 seconds). Please try submitting the job'
                response['status'] = STATUS_ERROR
                response['msg'] = err_out
        else:
            try:
                df = cursor.execute(query)
                connection.con.commit()
            except Exception as e:
                err_out = str(e).strip()
                if 'ORA-01013' in err_out:
                    err_out = 'Time Exceeded (30 seconds). Please try submitting this query as job'
                response['status'] = STATUS_ERROR
                response['msg'] = err_out

        # Stop the clock
        tt.cancel()
        # Close database connection
        cursor.close()
        connection.close()
    except Exception as e:
        response['status'] = STATUS_ERROR
        response['msg'] = str(e).strip()
        return response

    return response


def run_query(query, db, username, password, job_folder, filename, timeout=None):
    response = DEFAULT_RESPONSE
    if not os.path.exists(job_folder):
        os.mkdir(job_folder)
    jsonfile = os.path.join(job_folder, 'meta.json')
    try:
        t1 = time.time()
        # Open database connection
        try:
            connection = ea.connect(db, user=username, passwd=password)
            cursor = connection.cursor()
        except Exception as e:
            response['status'] = STATUS_ERROR
            response['msg'] = str(e).strip()
            with open(jsonfile, 'w') as fp:
                json.dump(response, fp)
            return response
        if timeout is not None:
            tt = threading.Timer(timeout, connection.con.cancel)
            tt.start()
        if query.lower().lstrip().startswith('select'):
            try:
                outfile = os.path.join(job_folder, filename)
                connection.query_and_save(query, outfile)
                if timeout is not None:
                    tt.cancel()
                t2 = time.time()
                files = glob.glob(job_folder + '/*')
                response['files'] = [os.path.basename(i) for i in files]
                response['sizes'] = [get_filesize(i) for i in files]
            except Exception as e:
                if timeout is not None:
                    tt.cancel()
                t2 = time.time()
                response['status'] = STATUS_ERROR
                response['msg'] = str(e).strip()
                raise
        else:
            try:
                cursor.execute(query)
                connection.con.commit()
                if timeout is not None:
                    tt.cancel()
                t2 = time.time()
            except Exception as e:
                if timeout is not None:
                    tt.cancel()
                t2 = time.time()
                response['status'] = STATUS_ERROR
                response['msg'] = str(e).strip()

        response['elapsed'] = t2 - t1
        with open(jsonfile, 'w') as fp:
            json.dump(response, fp, indent=2)
        cursor.close()
        connection.close()
        return response
    except Exception:
        raise
