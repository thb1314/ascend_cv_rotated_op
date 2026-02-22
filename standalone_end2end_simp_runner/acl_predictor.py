import numpy as np
import acl
import logging
import ctypes

logger = logging.getLogger("")


ACL_HOST = 1
ACL_ERROR_NONE = 0
ACL_MEM_MALLOC_NORMAL_ONLY = 2
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
NPY_BYTE = 1
ACL_FLOAT = 0
ACL_FLOAT16 = 1
ACL_INT32 = 3
ACL_UINT32 = 8
ACL_INT64 = 9
ACL_UINT64 = 10
ACL_UINT8 = 4
ACL_BOOL = 12
ACL_ERROR_REPEAT_INITIALIZE = 100002
ACL_ERROR_REPEAT_FINALIZE = 100037


class AclPredictor:
    """
    ACL Wrapper Class
    """

    def __init__(self, model_path='model_path', device_id=0, do_finalize=True):
        self.device_id = device_id
        self.model_path = model_path
        # whether to finalize when destroy, in case ACL is being used elsewhere
        self.do_finalize = do_finalize
        self.context = None
        self.model_id = None
        self.run_mode = None
        self.model_desc = None
        self.input_dataset = None
        self.input_buffer_ptrs = None
        self.input_buffer_sizes = None
        self.input_host_ptrs = None
        self.output_dataset = None
        self.output_info = None
        self.output_buffer_ptrs = None
        self.output_buffer_sizes = None
        self.input_names = None
        self.output_names = None
        self.is_dynamic_batch = False
        self.dynamic_batch_input_idx = None
        self.dynamic_batch_desc = None
        self.host_buffer = None
        self._closed = False
        self._init_resource()
        self._init_model()
        self._init_input()
        output_buffer_size = self._init_output()
        self.host_buffer = np.empty((int(output_buffer_size),), dtype=np.uint8)

    def _init_resource(self):
        ret = acl.init()
        # already initialized, ignore
        if ret != ACL_ERROR_REPEAT_INITIALIZE:
            self._check_ret_value("acl.init", ret)
        ret = acl.rt.set_device(self.device_id)
        self._check_ret_value("acl.rt.set_device", ret)
        self.context, ret = acl.rt.create_context(self.device_id)
        self._check_ret_value("acl.rt.create_context", ret)
        self.run_mode, ret = acl.rt.get_run_mode()
        self._check_ret_value("acl.rt.get_run_mode", ret)

    def _init_model(self):
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        self._check_ret_value("acl.mdl.load_from_file", ret)
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        self._check_ret_value("acl.mdl.get_desc", ret)

        try:
            in_num = acl.mdl.get_num_inputs(self.model_desc)
            names = []
            for i in range(in_num):
                try:
                    names.append(acl.mdl.get_input_name_by_index(self.model_desc, i))
                except Exception:
                    names.append(None)
            self.input_names = names
        except Exception:
            self.input_names = None

        try:
            out_num = acl.mdl.get_num_outputs(self.model_desc)
            names = []
            for i in range(out_num):
                try:
                    names.append(acl.mdl.get_output_name_by_index(self.model_desc, i))
                except Exception:
                    names.append(None)
            self.output_names = names
        except Exception:
            self.output_names = None
        dynamic_idx, ret = acl.mdl.get_input_index_by_name(self.model_desc, "ascend_mbatch_shape_data")
        if ACL_ERROR_NONE == ret:
            self.dynamic_batch_input_idx = dynamic_idx
            self.is_dynamic_batch = True
            batch_dic, ret = acl.mdl.get_dynamic_batch(self.model_desc)
            if ACL_ERROR_NONE == ret:
                self.dynamic_batch_desc = batch_dic

    def _init_input(self):
        input_num = acl.mdl.get_num_inputs(self.model_desc)
        self.input_buffer_ptrs = []
        self.input_buffer_sizes = []
        self.input_host_ptrs = [None for _ in range(input_num)]
        self.input_dataset = acl.mdl.create_dataset()
        for i in range(input_num):
            temp_buffer_size = acl.mdl.get_input_size_by_index(self.model_desc, i)
            input_ptr, ret = acl.rt.malloc(temp_buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY)
            self._check_ret_value("acl.rt.malloc", ret)
            data_buffer = acl.create_data_buffer(input_ptr, temp_buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(self.input_dataset, data_buffer)
            self._check_ret_value("acl.mdl.add_dataset_buffer", ret)
            self.input_buffer_ptrs.append(input_ptr)
            self.input_buffer_sizes.append(temp_buffer_size)

    def _init_output(self):
        output_num = acl.mdl.get_num_outputs(self.model_desc)
        self.output_dataset = acl.mdl.create_dataset()
        self.output_buffer_ptrs = []
        self.output_buffer_sizes = []
        output_buffer_size = 0

        def _dtype_size_bytes(datatype: int) -> int:
            if datatype == ACL_FLOAT:
                return 4
            if datatype == ACL_FLOAT16:
                return 2
            if datatype == ACL_INT32:
                return 4
            if datatype == ACL_UINT32:
                return 4
            if datatype == ACL_INT64:
                return 8
            if datatype == ACL_UINT64:
                return 8
            if datatype == ACL_UINT8:
                return 1
            if datatype == ACL_BOOL:
                return 1
            raise RuntimeError(f"unsurpport datatype {datatype}")

        for i in range(output_num):
            try:
                temp_buffer_size = int(acl.mdl.get_output_size_by_index(self.model_desc, i))
            except Exception:
                dims, ret = acl.mdl.get_output_dims(self.model_desc, i)
                self._check_ret_value("acl.mdl.get_output_dims", ret)
                datatype = acl.mdl.get_output_data_type(self.model_desc, i)
                shape = tuple(1 if int(d) < 0 else int(d) for d in dims["dims"])
                prod = 1
                for v in shape:
                    prod *= int(v)
                temp_buffer_size = int(prod) * _dtype_size_bytes(int(datatype))

            output_buffer_size = max(output_buffer_size, temp_buffer_size)
            temp_buffer, ret = acl.rt.malloc(int(temp_buffer_size), ACL_MEM_MALLOC_NORMAL_ONLY)
            dataset_buffer = acl.create_data_buffer(temp_buffer, temp_buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(self.output_dataset, dataset_buffer)
            self._check_ret_value("acl.mdl.add_dataset_buffer", ret)
            self.output_buffer_ptrs.append(temp_buffer)
            self.output_buffer_sizes.append(int(temp_buffer_size))
        output_info = []
        for i in range(output_num):
            dims, ret = acl.mdl.get_output_dims(self.model_desc, i)
            self._check_ret_value("acl.mdl.get_output_dims", ret)
            datatype = acl.mdl.get_output_data_type(self.model_desc, i)
            shape = tuple(1 if int(d) < 0 else int(d) for d in dims["dims"])
            output_info.append({"shape": shape, "type": datatype})
        self.output_info = output_info
        return output_buffer_size

    def _gen_input(self, input_data, batch=None):
        if not isinstance(input_data, list):
            input_data = [input_data]
        if self.is_dynamic_batch:
            if batch is None:
                raise RuntimeError("[ERROR] batch parameter can not be None when dynamic batch enabled")
            input_need_size = len(self.input_buffer_ptrs) - 1
            if len(input_data) != input_need_size:
                raise RuntimeError(f"[ERROR] the length of input_data is wrong, it should be {input_need_size}")
            if batch not in self.dynamic_batch_desc["batch"]:
                raise RuntimeError(f"[ERROR] [dynamic batch] {batch} is not in {self.dynamic_batch_desc['batch']}")
            input_data.append(np.array([batch], dtype=np.int32))

        for idx, (input_item, ptr) in enumerate(zip(input_data, self.input_buffer_ptrs)):
            if self.is_dynamic_batch and idx == self.dynamic_batch_input_idx:
                ret = acl.mdl.set_dynamic_batch_size(self.model_id,
                                                     self.input_dataset,
                                                     self.dynamic_batch_input_idx,
                                                     batch)
                self._check_ret_value("acl.mdl.set_dynamic_batch_size", ret)
            max_size = self.input_buffer_sizes[idx] if self.input_buffer_sizes is not None else input_item.nbytes
            if input_item.nbytes != max_size:
                raise RuntimeError(f"input[{idx}] nbytes mismatch: got={input_item.nbytes} expected={max_size}")
            src_ptr = input_item.ctypes.data_as(ctypes.c_void_p).value
            ret = acl.rt.memcpy(ptr, max_size, src_ptr, max_size, ACL_MEMCPY_HOST_TO_DEVICE)
            self._check_ret_value("acl.rt.memcpy", ret)

    def run(self, input_data, batch=None):
        ret = acl.rt.set_context(self.context)
        self._check_ret_value("acl.rt.set_context", ret)
        self._gen_input(input_data, batch)
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        self._check_ret_value("acl.mdl.execute", ret)
        result = self._output_dataset_to_numpy(batch)
        return result

    def _output_dataset_to_numpy(self, batch=None):
        output_result = []
        num = acl.mdl.get_dataset_num_buffers(self.output_dataset)
        if not self.is_dynamic_batch:
            batch = None
        for i in range(num):
            buffer_d = acl.mdl.get_dataset_buffer(self.output_dataset, i)
            scale = 1.0
            if batch is not None:
                max_batch = max(self.dynamic_batch_desc['batch'])
                try:
                    scale = batch / max_batch
                except ZeroDivisionError as e:
                    logger.error("max_batch can not be zero")
                    raise e

            data = acl.get_data_buffer_addr(buffer_d)
            base_size = int(self.output_buffer_sizes[i]) if self.output_buffer_sizes is not None else int(acl.get_data_buffer_size(buffer_d))
            size = int(round(base_size * scale))

            if self.run_mode == ACL_HOST:
                host_ptr = int(self.host_buffer.ctypes.data)
                ret = acl.rt.memcpy(host_ptr, size, data, size, ACL_MEMCPY_DEVICE_TO_HOST)
                self._check_ret_value("acl.rt.memcpy", ret)
                byte_buffer = self.host_buffer[:size].tobytes()
            else:
                byte_buffer = acl.util.ptr_to_bytes(data, size)
            try:
                dims, ret = acl.mdl.get_cur_output_dims(self.model_desc, i)
                self._check_ret_value("acl.mdl.get_cur_output_dims", ret)
                shape = [int(d) for d in dims["dims"]]
            except Exception:
                shape = list(self.output_info[i]["shape"])

            if batch is not None:
                max_batch = max(self.dynamic_batch_desc['batch'])
                for idx, v in enumerate(shape):
                    if int(v) == int(max_batch):
                        shape[idx] = int(batch)
                        break
            data_array = self._unpack_output_item(byte_buffer, tuple(shape), self.output_info[i]["type"])
            output_result.append(data_array)
        return output_result

    def _unpack_output_item(self, byte_buffer, shape, datatype):
        if datatype == ACL_FLOAT:
            np_type = np.float32
        elif datatype == ACL_FLOAT16:
            np_type = np.float16
        elif datatype == ACL_INT32:
            np_type = np.int32
        elif datatype == ACL_UINT32:
            np_type = np.uint32
        elif datatype == ACL_INT64:
            np_type = np.int64
        elif datatype == ACL_UINT64:
            np_type = np.uint64
        elif datatype == ACL_UINT8:
            np_type = np.uint8
        elif datatype == ACL_BOOL:
            np_type = np.bool_
        else:
            raise RuntimeError(f"unsurpport datatype {datatype}")
        return np.frombuffer(byte_buffer, dtype=np_type).reshape(shape)

    def _check_ret_value(self, message, ret, raise_err=True):
        if ret != ACL_ERROR_NONE:
            msg = f"{message} failed ret={ret}"
            if raise_err:
                raise RuntimeError(msg)
            logger.error(msg)

    def __del__(self):
        self.close()

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self._release_dataset()
        if self.input_buffer_ptrs:
            for ptr in self.input_buffer_ptrs:
                ret = acl.rt.free(ptr)
                self._check_ret_value("acl.rt.free", ret, False)
        self.input_buffer_ptrs = None
        self.input_buffer_sizes = None
        self.input_host_ptrs = None
        self.host_buffer = None
        if self.model_id:
            ret = acl.mdl.unload(self.model_id)
            self._check_ret_value("acl.mdl.unload", ret, False)
        self.model_id = None
        if self.model_desc:
            ret = acl.mdl.destroy_desc(self.model_desc)
            self._check_ret_value("acl.mdl.destroy_desc", ret, False)
        self.model_desc = None
        if self.context:
            ret = acl.rt.destroy_context(self.context)
            self._check_ret_value("acl.rt.destroy_context", ret, False)
        self.context = None
        if self.do_finalize:
            ret = acl.rt.reset_device(self.device_id)
            self._check_ret_value("acl.rt.reset_device", ret, False)
            ret = acl.finalize()
            # already finalized, ignore
            if ret != ACL_ERROR_REPEAT_FINALIZE:
                self._check_ret_value("acl.finalize", ret, False)
        self.device_id = None

    def _release_dataset(self):
        for dataset in [self.input_dataset, self.output_dataset]:
            if not dataset:
                continue
            buf_num = acl.mdl.get_dataset_num_buffers(dataset)
            for i in range(buf_num):
                data_buf = acl.mdl.get_dataset_buffer(dataset, i)
                if data_buf:
                    ret = acl.destroy_data_buffer(data_buf)
                    self._check_ret_value("acl.destroy_data_buffer", ret, False)
            ret = acl.mdl.destroy_dataset(dataset)
            self._check_ret_value("acl.mdl.destroy_dataset", ret, False)
        self.input_dataset = None
        self.output_dataset = None
